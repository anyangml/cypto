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

        比较当前挂单与计划挂单，最小化撤单/补单操作。
        """
        # 从策略配置读取最小下单量，避免硬编码
        min_qty = float(self.strategy_cfg.grid_engine.min_order_qty)

        def price_key(p: float) -> float:
            return round(float(p), 2)

        current_buys: Dict[float, dict] = {
            price_key(o["price"]): o
            for o in open_orders
            if o["side"] == "buy"
        }
        current_sells: Dict[float, dict] = {
            price_key(o["price"]): o
            for o in open_orders
            if o["side"] == "sell"
        }

        plan_buy_orders: List[dict] = plan.get("buy_orders", [])
        plan_sell_orders: List[dict] = plan.get("sell_orders", [])
        plan_buy_prices = {price_key(p["price"]) for p in plan_buy_orders}
        plan_sell_prices = {price_key(p["price"]) for p in plan_sell_orders}

        logger.info(
            "Diffing: 当前买单 %d vs 计划 %d | 当前卖单 %d vs 计划 %d",
            len(current_buys), len(plan_buy_prices),
            len(current_sells), len(plan_sell_prices),
        )

        # 1. 撤销不再需要的订单
        for p, order in current_buys.items():
            if p not in plan_buy_prices:
                logger.info("撤销多余买单: %.2f", p)
                try:
                    self.exchange.cancel_order(order["id"])
                except Exception as e:
                    logger.warning("撤销买单失败 %.2f: %s", p, e)

        for p, order in current_sells.items():
            if p not in plan_sell_prices:
                logger.info("撤销多余卖单: %.2f", p)
                try:
                    self.exchange.cancel_order(order["id"])
                except Exception as e:
                    logger.warning("撤销卖单失败 %.2f: %s", p, e)

        # 2. 补挂缺失的订单
        buy_enabled = plan.get("buy_grid_enabled", True)
        sell_enabled = plan.get("sell_grid_enabled", True)

        if buy_enabled:
            for item in plan_buy_orders:
                p_key = price_key(item["price"])
                if p_key not in current_buys and float(item["price"]) < current_price:
                    try:
                        self.exchange.create_limit_order("buy", float(item["price"]), min_qty)
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error("补挂买单失败 %.2f: %s", item["price"], e)

        if sell_enabled:
            for item in plan_sell_orders:
                p_key = price_key(item["price"])
                if p_key not in current_sells and float(item["price"]) > current_price:
                    try:
                        self.exchange.create_limit_order("sell", float(item["price"]), min_qty)
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error("补挂卖单失败 %.2f: %s", item["price"], e)


if __name__ == "__main__":
    executor = GridStrategyExecutor()
    executor.run()
