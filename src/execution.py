"""
execution.py - 实盘/Testnet 策略执行器

负责调度：
  1. 获取市场数据 (Data)
  2. 计算市场状态 (Regime)
  3. 生成网格计划 (Grid)
  4. 执行订单同步 (Execution)
  5. 风险控制中断 (Risk Control)
  6. 成交记录持久化 (Database)
"""

import time
from typing import Dict, List

from src.config import Config
from src.strategy_config import load_strategy_config
from src.regime_filter import RegimeFilter, MarketRegime
from src.grid_engine import GridEngine
from src.risk_control import RiskManager, RiskStatus
from src.exchange import ExchangeClient
from src.database import DatabaseHandler
from src.logger import setup_logger

logger = setup_logger("execution")


class GridStrategyExecutor:
    """实盘/Testnet 网格策略执行器。"""

    def __init__(self):
        self.cfg = Config()
        self.strategy_cfg = load_strategy_config()

        # 初始化各功能组件
        self.exchange = ExchangeClient(self.cfg)
        self.rf = RegimeFilter(
            self.strategy_cfg.regime_filter,
            self.strategy_cfg.position_sizing.tiers,
        )
        self.ge = GridEngine(self.strategy_cfg.grid_engine)
        self.rm = RiskManager(
            self.strategy_cfg.risk_control,
            self.strategy_cfg.position_sizing.fuse_conditions,
        )
        self.db = DatabaseHandler()

        self.symbol = self.cfg.symbol
        self.timeframe = self.cfg.timeframe

        # 最小挂单名义价值 (USDT)，来自策略配置；默认 5.0 防止 Binance 拒单
        self._min_notional_value: float = getattr(
            self.strategy_cfg, "min_order_value", 5.0
        )

        # 上次同步成交记录的时间戳（秒）
        self._last_trade_sync_ts: float = 0.0

        logger.info("GridStrategyExecutor 初始化完成")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self):
        """主循环：每 30 秒执行一次策略步骤。"""
        logger.info("策略引擎启动...")

        while True:
            try:
                self.step()
                wait_sec = 30
                logger.info("等待 %d 秒后进入下一轮检测...", wait_sec)
                time.sleep(wait_sec)

            except KeyboardInterrupt:
                logger.warning("收到停止信号，正在退出...")
                break
            except Exception as e:
                logger.error("主循环发生未知错误: %s", e, exc_info=True)
                time.sleep(10)

    # ------------------------------------------------------------------
    # 单次执行步骤
    # ------------------------------------------------------------------

    def step(self):
        """单次策略执行逻辑。"""
        logger.info("=== 开始新一轮策略计算 ===")

        # ── 1. 获取账户余额与 K 线数据 ──────────────────────────────
        raw_balance = self.exchange.fetch_balance()
        # fetch_balance 返回 {asset: float_total}，不是嵌套 dict
        usdt_total = float(raw_balance.get("USDT", 0.0))
        btc_total = float(raw_balance.get("BTC", 0.0))

        df = self.exchange.fetch_ohlcv(
            since_days=self.strategy_cfg.regime_filter.hurst_period * 2 // 24 + 1
        )
        if df.empty:
            logger.warning("获取不到 K 线数据，跳过本轮")
            return

        current_price = float(df["close"].iloc[-1])
        total_equity = usdt_total + btc_total * current_price

        # ── 1.5 风控检查 ─────────────────────────────────────────────
        try:
            fr = self.exchange.fetch_funding_rate()
            oi = self.exchange.fetch_open_interest()
            risk = self.rm.check(current_oi=oi, current_funding_rate=fr)

            if risk.status in (RiskStatus.FUSE, RiskStatus.PAUSE):
                logger.error(
                    "风控触发: %s | Status: %s", risk.reason, risk.status
                )
                self.exchange.cancel_all_orders()
                return
        except Exception as e:
            logger.warning("风控数据获取失败，跳过检查: %s", e)

        # ── 2. Regime Filter ─────────────────────────────────────────
        try:
            regime_res = self.rf.get_market_regime(df)
        except ValueError as e:
            logger.warning("Regime 计算失败(数据不足?): %s", e)
            return

        logger.info("当前 Regime: %s", regime_res)

        # ── 3. 决策分支 ──────────────────────────────────────────────
        if regime_res.regime == MarketRegime.FUSE:
            logger.warning("触发熔断状态！正在撤销所有订单...")
            self.exchange.cancel_all_orders()
            return

        if regime_res.regime == MarketRegime.TREND:
            logger.info("趋势状态，暂停网格开单，撤销所有挂单")
            self.exchange.cancel_all_orders()
            return

        # Regime == RANGE
        if regime_res.position_ratio <= 0:
            logger.info("震荡但仓位建议为 0 (高 Hurst)，撤单观望")
            self.exchange.cancel_all_orders()
            return

        # ── 4. 生成网格计划 ──────────────────────────────────────────
        grid_plan = self.ge.calculate_grid(
            df,
            total_balance=total_equity,
            position_ratio=regime_res.position_ratio,
        )
        logger.info(
            "网格计划: Range=[%.2f, %.2f], Mid=%.2f",
            grid_plan["lower_boundary"],
            grid_plan["upper_boundary"],
            grid_plan["mid_price"],
        )

        # ── 5. 订单同步 (Diffing) ────────────────────────────────────
        try:
            open_orders = self.exchange.fetch_open_orders()
        except Exception as e:
            logger.error("无法获取挂单，跳过同步: %s", e)
            return

        self._sync_orders(grid_plan, open_orders, current_price)

        # ── 6. 定期同步成交记录到数据库 (每 5 分钟) ─────────────────
        now = time.monotonic()
        if now - self._last_trade_sync_ts > 300:
            self._sync_trades()
            self._last_trade_sync_ts = now

        logger.info("---------------- Step End ----------------")

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _sync_trades(self):
        """从交易所拉取最新成交记录并写入本地数据库。"""
        try:
            # 使用 ExchangeClient 封装的 _safe_call 保证重试
            trades = self.exchange._safe_call(
                self.exchange._exchange.fetch_my_trades,
                self.symbol,
                limit=50,
            )
            if trades:
                self.db.record_trades(trades)
        except Exception as e:
            logger.warning("同步成交记录失败: %s", e)

    def _sync_orders(
        self,
        plan: dict,
        open_orders: list,
        current_price: float,
    ):
        """
        智能订单同步 (Diffing)。
        比较当前挂单与计划挂单，进行最小化调整，并根据资金利用率动态计算下单量。
        """
        min_notional_value = self._min_notional_value

        # 1. 精度与 Key 处理
        def price_key(p: float) -> str:
            # 使用交易所精度规则处理价格 Key，避免浮点误差
            return str(self.exchange.price_to_precision(self.symbol, p))

        current_buys: Dict[str, dict] = {}
        for o in open_orders:
            if o["side"] == "buy":
                current_buys[price_key(o["price"])] = o

        current_sells: Dict[str, dict] = {}
        for o in open_orders:
            if o["side"] == "sell":
                current_sells[price_key(o["price"])] = o

        plan_buy_orders = plan.get("buy_orders", [])
        plan_sell_orders = plan.get("sell_orders", [])
        
        # 预计算所有计划单的价格 Key
        plan_buy_keys = {price_key(p["price"]) for p in plan_buy_orders}
        plan_sell_keys = {price_key(p["price"]) for p in plan_sell_orders}

        logger.info(
            "Diffing: 买单 %d(curr) vs %d(plan) | 卖单 %d(curr) vs %d(plan)",
            len(current_buys), len(plan_buy_keys),
            len(current_sells), len(plan_sell_keys),
        )

        # 2. 撤销不再需要的订单
        for p_key, order in current_buys.items():
            if p_key not in plan_buy_keys:
                logger.info("撤销多余买单: %s", p_key)
                try: self.exchange.cancel_order(order["id"])
                except Exception as e: logger.warning("撤单失败 %s: %s", p_key, e)

        for p_key, order in current_sells.items():
            if p_key not in plan_sell_keys:
                logger.info("撤销多余卖单: %s", p_key)
                try: self.exchange.cancel_order(order["id"])
                except Exception as e: logger.warning("撤单失败 %s: %s", p_key, e)

        # 3. 计算资金分配 (Capital Allocation)
        # 获取当前可用余额
        balance = self.exchange.fetch_balance()
        usdt_available = balance.get("USDT", 0.0)
        btc_available = balance.get("BTC", 0.0)
        
        # 资金利用率 (由 Regime/GridEngine 决定)
        utilization = plan.get("capital_utilization", 1.0)
        # 目标挂单总金额 = 当前余额 * 利用率
        # 注意：这里的余额包含已挂单资金吗？fetch_balance 通常返回 total=free+used
        # 如果是 Diffing 模式，已挂单资金已经在 used 中。
        # 简单起见，我们计算每格应该分配多少，然后挂新单
        
        # 估算总权益用于计算每格资金
        # (Total USDT + BTC Value) * Utilization / Total Grids
        # 这里简化为：将 可用(Free) 资金分配给 新增(Missing) 订单
        
        # 统计缺失的订单数
        missing_buys = [p for p in plan_buy_orders if price_key(p["price"]) not in current_buys]
        missing_sells = [p for p in plan_sell_orders if price_key(p["price"]) not in current_sells]
        
        # 如果没有缺失，直接返回
        if not missing_buys and not missing_sells:
            return

        # 资金分配策略：
        # 将 *当前可用余额* 均分给 *缺失的订单*
        # 这是一种保守策略，确保有钱挂单
        # 如果 utilization 很低，我们应该只用部分可用余额
        
        # 修正：我们需要考虑到 "Total Target Investment"
        # 但 Diffing 比较复杂。简单的做法：
        # Use (Available Balance * Utilization) / (Missing Orders Count + Safety Buffer)
        
        # 挂买单 (USDT)
        if missing_buys:
            # 这里的可用余额是 Free Balance
            # 假设我们只用 Free Balance 的一部分
            # 如果 utilization=1.0, 用所有 Free? 是的。
            # 安全系数 0.98 防止精度问题
            budget = usdt_available * utilization * 0.98
            amount_per_order_usdt = budget / len(missing_buys)
            
            if amount_per_order_usdt < min_notional_value:
                logger.warning(f"资金不足，无法补齐所有买单 (PerOrder {amount_per_order_usdt:.2f} < 5.0)")
                # 尝试挂一部分? 或者跳过
                sorted_buys = sorted(missing_buys, key=lambda x: x['price'], reverse=True) # 优先挂靠近现价的
                # 重新计算能挂几个
                num_can_afford = int(budget // min_notional_value)
                missing_buys = sorted_buys[:num_can_afford]
                if missing_buys:
                    amount_per_order_usdt = budget / len(missing_buys)
            
            for item in missing_buys:
                price = item["price"]
                # 再次检查价格逻辑 (买单 < 现价)
                if float(price) >= current_price: continue
                
                qty = amount_per_order_usdt / float(price)
                
                # 精度修正
                final_qty = self.exchange.amount_to_precision(self.symbol, qty)
                final_price = self.exchange.price_to_precision(self.symbol, float(price))
                
                # 最小下单量检查 (qty * price >= 5)
                if final_qty * float(final_price) < min_notional_value:
                    continue
                    
                try:
                    self.exchange.create_limit_order("buy", final_price, final_qty)
                    time.sleep(0.1)
                except Exception as e:
                    logger.error("补挂买单失败 %s: %s", final_price, e)

        # 挂卖单 (BTC)
        if missing_sells:
            budget_btc = btc_available * utilization * 0.98
            qty_per_order = budget_btc / len(missing_sells)
            
            # 检查最小名义价值
            # 估算 value = qty * price
            # 如果 qty 太小导致 value < 5，则减少挂单数
            # 取第一笔卖单价格估算
            est_price = float(missing_sells[0]["price"])
            if qty_per_order * est_price < min_notional_value:
                # 削减订单
                min_btc_req = min_notional_value / est_price
                num_can_afford = int(budget_btc // min_btc_req)
                sorted_sells = sorted(missing_sells, key=lambda x: x['price']) # 优先挂靠近现价的 (低价)
                missing_sells = sorted_sells[:num_can_afford]
                if missing_sells:
                    qty_per_order = budget_btc / len(missing_sells)
            
            for item in missing_sells:
                price = item["price"]
                if float(price) <= current_price: continue

                final_qty = self.exchange.amount_to_precision(self.symbol, qty_per_order)
                final_price = self.exchange.price_to_precision(self.symbol, float(price))

                if final_qty * float(final_price) < min_notional_value:
                    continue
                    
                try:
                    self.exchange.create_limit_order("sell", final_price, final_qty)
                    time.sleep(0.1)
                except Exception as e:
                    logger.error("补挂卖单失败 %s: %s", final_price, e)


if __name__ == "__main__":
    executor = GridStrategyExecutor()
    executor.run()
