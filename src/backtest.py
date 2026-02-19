"""
src/backtest.py - 回测引擎（Feed 架构版）

核心改进：
1. 通过 DataFeed 协议消费数据，Backtest / Live 均可接入（零改动引擎层）
2. Tick-by-tick 撮合：价格按 waypoint 路径运动，逐段击穿网格层
   · 避免"同根 K 线扫单全成交"的虚假精度
   · 每笔成交有唯一时间戳（bar_ts + waypoint_offset），History 不再相同时刻
3. 订单使用自增 ID，彻底避免同价位重建网格时 ID 冲突
4. 所有可配置参数集中在 __init__ 或 StrategyConfig，无 hardcode

撮合逻辑说明（与 Live 对齐）：
  Live 场景: 实盘网格挂的是限价单，交易所在价格穿越时成交一笔
             → 同一个 WebSocket 推送周期内最多一笔成交
  回测场景: BacktestFeed 将每根 K 线拆成 4 个 waypoint，
             价格从 A 移动到 B 时，只成交路径内的网格层
             → 与限价单逐层成交行为吻合
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, List, Optional

from src.config import Config
from src.strategy_config import load_strategy_config
from src.regime_filter import RegimeFilter, MarketRegime, RegimeResult
from src.grid_engine import GridEngine
from src.logger import setup_logger
from src.feed import BarData, TickData, DataFeed, BacktestFeed

logger = setup_logger("backtest")


class BacktestEngine:
    """
    事件驱动回测引擎。

    通过 DataFeed 协议接收数据，任何实现了 iter_bars() 的 Feed 均可接入:
        engine = BacktestEngine()
        engine.run(BacktestFeed(df, timeframe="1m"))
        # 未来:
        engine.run(LiveFeed(symbol="BTC/USDT", timeframe="1m"))

    参数（来自 StrategyConfig，无 hardcode）：
        strategy_cfg.backtest.initial_capital   初始权益 (USDT)
        strategy_cfg.backtest.fee_rate          手续费率
        strategy_cfg.backtest.min_order_value   最小订单名义价值
        strategy_cfg.backtest.initial_pos_ratio 初始 BTC 仓位占比
    """

    def __init__(self, strategy_cfg=None) -> None:
        self.strategy_cfg = strategy_cfg or load_strategy_config()
        self.rf = RegimeFilter(
            self.strategy_cfg.regime_filter,
            self.strategy_cfg.position_sizing.tiers,
        )
        self.ge = GridEngine(self.strategy_cfg.grid_engine)

        # 回测账户参数：从配置读取，提供合理默认值
        bt_cfg = getattr(self.strategy_cfg, "backtest", None) or {}
        self._initial_capital:   float = getattr(bt_cfg, "initial_capital",   10_000.0)
        self._fee_rate:          float = getattr(bt_cfg, "fee_rate",           0.001)
        self._min_order_value:   float = getattr(bt_cfg, "min_order_value",    5.0)
        self._initial_pos_ratio: float = getattr(bt_cfg, "initial_pos_ratio",  0.5)

        # 运行时状态（每次 run() 前重置）
        self.balance_usdt: float = 0.0
        self.balance_btc:  float = 0.0
        self.orders:       List[Dict] = []
        self.trades:       List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.last_grid_mid: Optional[float] = None
        self._order_seq:   int = 0   # 全局自增 ID，确保每笔挂单 ID 唯一

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(self, feed: DataFeed, df: pd.DataFrame) -> Dict:
        """
        运行回测主流程。

        Args:
            feed: 实现了 DataFeed 协议的数据源（BacktestFeed 或未来 LiveFeed）
            df:   完整 OHLCV DataFrame（供 RegimeFilter 预计算和 GridEngine 使用）

        Returns:
            包含 equity_curve / trades / metrics 的报告字典
        """
        if df.empty:
            logger.warning("空 DataFrame，回测终止")
            return {}

        self._reset(df)

        # --- 预计算所有 Regime（性能优化，避免滚动窗口重复计算）---
        logger.info("正在全量计算 Regime 信号...")
        regime_results: List[Optional[RegimeResult]] = self.rf.scan_dataframe_full(df)

        # 预热期：需要足够数据给 MA200 / ATR 等指标
        warmup_bars = max(self.ge.config.ma_long_period, 200)

        for bar_idx, (bar, ticks) in enumerate(feed.iter_bars()):
            if bar_idx < warmup_bars:
                continue

            # A. 订单撮合（tick-by-tick，模拟价格路径逐层击穿网格）
            self._match_with_ticks(ticks)

            # B. 策略逻辑（在收线后执行，与 Live WebSocket is_closed=True 对齐）
            regime_res = regime_results[bar_idx] if bar_idx < len(regime_results) else None
            if not regime_res:
                self._record_snapshot(bar.timestamp, bar.close, "N/A")
                continue

            self._run_strategy(bar, df, bar_idx, regime_res)

            # C. 权益快照
            self._record_snapshot(bar.timestamp, bar.close, regime_res.regime.value)

        return self._generate_report()

    # ------------------------------------------------------------------
    # 状态重置
    # ------------------------------------------------------------------

    def _reset(self, df: pd.DataFrame) -> None:
        """初始化账户状态，按配置比例分配初始仓位。"""
        start_price = float(df.iloc[0]["close"])
        self.balance_usdt = self._initial_capital * self._initial_pos_ratio
        self.balance_btc  = (self._initial_capital * (1 - self._initial_pos_ratio)) / start_price
        self.orders       = []
        self.trades       = []
        self.equity_curve = []
        self.last_grid_mid = None
        self._order_seq   = 0
        logger.info(
            "Backtest 初始化 | %.2f USDT + %.6f BTC @ %.2f",
            self.balance_usdt, self.balance_btc, start_price,
        )

    # ------------------------------------------------------------------
    # 撮合引擎（核心精度改进）
    # ------------------------------------------------------------------

    def _match_with_ticks(self, ticks: List[TickData]) -> None:
        """
        Tick-by-tick 订单撮合。

        价格从 prev → curr 运动时，只触发路径内的网格层：
          · 下行（curr < prev）→ 按价格从高到低逐个成交买单
          · 上行（curr > prev）→ 按价格从低到高逐个成交卖单

        这与限价单的真实成交行为吻合：
          · 挂在 66,500 的买单只在价格从 67,000 下穿 66,500 时成交
          · 不会因为 K 线最低价是 65,000 就与挂在 65,100 的买单同时成交
        """
        if not self.orders:
            return

        executed_ids: set = set()
        prev_price = ticks[0].price

        for tick in ticks[1:]:
            curr_price = tick.price
            ts         = tick.timestamp

            if curr_price < prev_price:
                # 价格下行 → 成交区间 [curr_price, prev_price] 内的买单（从高到低）
                candidates = [
                    o for o in self.orders
                    if o["side"] == "buy"
                    and o["id"]  not in executed_ids
                    and curr_price <= o["price"] <= prev_price
                ]
                for order in sorted(candidates, key=lambda x: -x["price"]):
                    if self._execute_trade(order, ts):
                        executed_ids.add(order["id"])

            elif curr_price > prev_price:
                # 价格上行 → 成交区间 [prev_price, curr_price] 内的卖单（从低到高）
                candidates = [
                    o for o in self.orders
                    if o["side"] == "sell"
                    and o["id"]  not in executed_ids
                    and prev_price <= o["price"] <= curr_price
                ]
                for order in sorted(candidates, key=lambda x: x["price"]):
                    if self._execute_trade(order, ts):
                        executed_ids.add(order["id"])

            prev_price = curr_price

        # 移除已成交的挂单
        self.orders = [o for o in self.orders if o["id"] not in executed_ids]

    # ------------------------------------------------------------------
    # 策略逻辑
    # ------------------------------------------------------------------

    def _run_strategy(
        self,
        bar: BarData,
        df: pd.DataFrame,
        bar_idx: int,
        regime_res: RegimeResult,
    ) -> None:
        """
        收线后触发策略决策（与 Live is_closed=True 对齐）。

        Regime 状态机：
          TREND / FUSE → 清空所有挂单，停止新开网格
          RANGE        → 检查是否需要重置网格（无挂单 或 价格偏离超阈值）
        """
        current_price = bar.close
        regime_status = regime_res.regime

        if regime_status in (MarketRegime.FUSE, MarketRegime.TREND):
            if self.orders:
                self.orders.clear()
                self.last_grid_mid = None

        elif regime_status == MarketRegime.RANGE:
            should_reset = not self.orders  # 无挂单时必须重置

            if self.last_grid_mid:
                drift = abs(current_price - self.last_grid_mid) / self.last_grid_mid
                grid_reset_threshold = getattr(
                    self.strategy_cfg, "grid_reset_drift_pct", 0.015
                )
                if drift > grid_reset_threshold:
                    should_reset = True

            # 风控：仓位比例为 0 时强制清空
            if regime_res.position_ratio <= 0:
                self.orders.clear()
                self.last_grid_mid = None
                return

            if should_reset:
                self.orders.clear()
                total_equity = self.balance_usdt + self.balance_btc * current_price
                calc_window  = df.iloc[max(0, bar_idx - 300): bar_idx + 1]
                grid_plan    = self.ge.calculate_grid(
                    calc_window,
                    total_balance=total_equity,
                    position_ratio=regime_res.position_ratio,
                )
                self.last_grid_mid = grid_plan["mid_price"]
                self._place_orders(grid_plan, current_price)

    # ------------------------------------------------------------------
    # 交易执行
    # ------------------------------------------------------------------

    def _execute_trade(self, order: Dict, ts: int) -> bool:
        """执行一笔交易，更新余额，记录快照（含持仓前后变化）。"""
        price = order["price"]
        qty   = order["qty"]
        cost  = price * qty
        fee   = cost * self._fee_rate

        if order["side"] == "buy":
            if self.balance_usdt < cost:
                return False
            pre_usdt, pre_btc = self.balance_usdt, self.balance_btc
            self.balance_usdt -= cost
            self.balance_btc  += qty * (1 - self._fee_rate)

        elif order["side"] == "sell":
            if self.balance_btc < qty:
                return False
            pre_usdt, pre_btc = self.balance_usdt, self.balance_btc
            self.balance_btc  -= qty
            self.balance_usdt += cost * (1 - self._fee_rate)

        else:
            return False

        self.trades.append({
            "time":       ts,
            "side":       order["side"],
            "price":      price,
            "qty":        qty,
            "fee":        fee,
            "quote_qty":  cost,
            # 持仓快照（交易前）
            "pre_usdt":   round(pre_usdt,          4),
            "pre_btc":    round(pre_btc,            6),
            # 持仓变化量
            "delta_usdt": round(self.balance_usdt - pre_usdt, 4),
            "delta_btc":  round(self.balance_btc  - pre_btc,  6),
        })
        return True

    def _place_orders(self, plan: Dict, current_price: float) -> None:
        """根据网格计划生成挂单，资金按网格层均分。"""
        utilization       = plan.get("capital_utilization", 1.0)
        target_usdt       = self.balance_usdt * utilization
        target_btc        = self.balance_btc  * utilization

        # 获取网格锁定状态（来自 GridEngine 的保护逻辑）
        buy_enabled  = plan.get("buy_grid_enabled", True)
        sell_enabled = plan.get("sell_grid_enabled", True)

        # 买单：价格须低于当前价 且 买单开启
        valid_buys  = []
        if buy_enabled:
            valid_buys = [o for o in plan["buy_orders"] if o["price"] < current_price]
            
        # 卖单：价格须高于当前价 且 卖单开启
        valid_sells = []
        if sell_enabled:
            valid_sells = [o for o in plan["sell_orders"] if o["price"] > current_price]

        # 1. 下买单 (USDT -> BTC)
        if valid_buys and target_usdt > self._min_order_value:
            amount_per = target_usdt / len(valid_buys)
            for bo in valid_buys:
                qty = amount_per / bo["price"]
                # 必须满足最小下单量和最小名义价值
                if amount_per >= self._min_order_value and qty >= self.ge.config.min_order_qty:
                    self._order_seq += 1
                    self.orders.append({
                        "id":    self._order_seq,
                        "side":  "buy",
                        "price": bo["price"],
                        "qty":   qty,
                    })

        # 2. 下卖单 (BTC -> USDT)
        if valid_sells and target_btc > 0:
            qty_per = target_btc / len(valid_sells)
            for so in valid_sells:
                notional = qty_per * so["price"]
                # 必须满足最小下单量和最小名义价值
                if notional >= self._min_order_value and qty_per >= self.ge.config.min_order_qty:
                    self._order_seq += 1
                    self.orders.append({
                        "id":    self._order_seq,
                        "side":  "sell",
                        "price": so["price"],
                        "qty":   qty_per,
                    })

    # ------------------------------------------------------------------
    # 权益快照 & 报告生成
    # ------------------------------------------------------------------

    def _record_snapshot(self, ts_sec: int, price: float, regime_label: str) -> None:
        total = self.balance_usdt + self.balance_btc * price
        self.equity_curve.append({
            "time":   ts_sec,
            "value":  round(total, 2),
            "regime": regime_label,
        })

    def _generate_report(self) -> Dict:
        if not self.equity_curve:
            return {}

        final_val   = self.equity_curve[-1]["value"]
        initial_val = self._initial_capital
        ret_pct     = (final_val - initial_val) / initial_val

        # Max Drawdown
        peak   = self.equity_curve[0]["value"]
        max_dd = 0.0
        for snap in self.equity_curve:
            if snap["value"] > peak:
                peak = snap["value"]
            dd = (peak - snap["value"]) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return {
            "equity_curve": self.equity_curve,
            "trades":       self.trades,
            "metrics": {
                "total_return_pct":  round(ret_pct * 100, 2),
                "final_value":       round(final_val,      2),
                "max_drawdown_pct":  round(max_dd * 100,   2),
                "trade_count":       len(self.trades),
                # 最终持仓
                "final_usdt":        round(self.balance_usdt, 2),
                "final_btc":         round(self.balance_btc,  6),
            },
        }
