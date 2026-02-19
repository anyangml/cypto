"""
src/grid_engine.py - 自适应网格逻辑引擎 (v2.1)

负责计算网格中轴线、上下边界、资金利用率以及生成具体的挂单列表。
支持以下核心特性：
    - MA200/MA20 多空背景与资金分配
    - ATR 动态边界
    - 布林带 2.5σ + Bias 超买保护
    - 按实际资金计算每层挂单 qty（修复 qty=0 占位符 Bug）
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field

from src.logger import setup_logger

logger = setup_logger("grid_engine")


class GridEngineConfig(BaseModel):
    """自适应网格引擎参数。"""

    # --- 基准线 ---
    ma_long_period: int   = Field(default=200, ge=10)
    ma_mid_period:  int   = Field(default=20,  ge=5)
    ma200_slope_lookback: int = Field(default=10, ge=1, description="计算 MA200 斜率的回看周期")
    ma200_slope_threshold: float = Field(default=0.0, description="MA200 斜率低于此值视为下行")

    # --- 资金利用率 ---
    capital_above_ma_long: float = Field(default=0.8, gt=0, le=1.0)
    capital_below_ma_long: float = Field(default=0.3, gt=0, le=1.0)

    # --- 动态区间 (ATR) ---
    atr_band_multiplier: float = Field(default=3.0, gt=0)
    atr_period:          int   = Field(default=14,  ge=2)

    # --- 超买保护 ---
    overbought_bb_std:          float = Field(default=2.5, gt=0)
    overbought_bb_period:       int   = Field(default=20,  ge=2)
    bias_overbought_threshold:  float = Field(default=0.05, gt=0)

    # --- 网格设置 ---
    grid_levels:         int   = Field(default=10,    ge=2)
    min_grid_spacing_pct: float = Field(default=0.003, gt=0)
    max_grid_spacing_pct: float = Field(default=0.05,  gt=0)
    min_order_qty:       float = Field(default=0.0004, gt=0,  description="最小单笔下单量 (BTC)")
    min_order_value:     float = Field(default=5.0,    gt=0,  description="最小单笔名义价值 (USDT)")

    # 资金分配：active_capital 中有多少比例分配给买方（USDT），剩余估算为 BTC 分配卖方
    usdt_ratio: float = Field(default=0.5, gt=0, le=1.0, description="active_capital 中买方 USDT 占比")
    
    # --- 偏置网格 / 持仓倾斜 (Biased Grid) ---
    dynamic_usdt_ratio_enabled: bool = Field(default=False, description="是否在空头背景下自动降低买单权重")
    bear_market_usdt_ratio:      float = Field(default=0.2, gt=0, le=1.0, description="空头背景下的 usdt_ratio")

    # --- 重置逻辑 ---
    grid_reset_drift_pct: float = Field(default=0.03, gt=0, description="价格偏离中轴多少比例时重置网格")


class GridEngine:
    """自适应网格计算引擎。"""

    def __init__(self, config: Optional[GridEngineConfig] = None) -> None:
        self.config = config or GridEngineConfig()
        logger.info("GridEngine 初始化完成 | 配置: %s", self.config)

    def calculate_grid(
        self,
        df: pd.DataFrame,
        total_balance: float,
        position_ratio: float = 1.0,
    ) -> dict:
        """
        根据最新市场数据计算网格计划，含每层实际挂单数量。

        Args:
            total_balance:  当前总权益（USDT 折算）。
            position_ratio: 来自 RegimeFilter 分级响应的仓位比例（0~1）。

        Returns:
            dict，包含:
                mid_price         网格中轴价 (MA20)
                upper_boundary    网格上限
                lower_boundary    网格下限
                capital_utilization  综合资金利用率
                active_capital    实际激活资金 (USDT)
                buy_orders        [{price, qty}, ...]  qty 已按资金计算
                sell_orders       [{price, qty}, ...]
                overbought_triggered  bool
                bias_pct          MA20 偏离度 (%)
                is_bull_market    价格 > MA200
                buy_grid_enabled  bool
                sell_grid_enabled bool
        """
        if len(df) < self.config.ma_long_period:
            raise ValueError(
                f"数据不足：需要至少 {self.config.ma_long_period} 根 K 线计算基准线"
            )

        latest_close = float(df["close"].iloc[-1])

        # 1. 基准线 (MA200, MA20)
        ma_long = float(df["close"].rolling(self.config.ma_long_period).mean().iloc[-1])
        ma_mid  = float(df["close"].rolling(self.config.ma_mid_period).mean().iloc[-1])

        # 2. 资金利用率
        is_bull = latest_close > ma_long
        capital_base   = self.config.capital_above_ma_long if is_bull else self.config.capital_below_ma_long
        capital_util   = capital_base * position_ratio          # 综合利用率
        active_capital = total_balance * capital_util           # 实际激活资金 (USDT)

        # 3. 动态边界 (ATR)
        atr   = self._calc_atr(df, self.config.atr_period)
        upper = latest_close + self.config.atr_band_multiplier * atr
        lower = latest_close - self.config.atr_band_multiplier * atr

        # 4. 超买保护 (BB 2.5σ + Bias)
        bias      = (latest_close - ma_mid) / ma_mid
        bb_upper  = self._calc_bb_upper(df, self.config.overbought_bb_period, self.config.overbought_bb_std)
        overbought_triggered = (
            latest_close >= bb_upper and bias > self.config.bias_overbought_threshold
        )

        # 5. 下行趋势判断 (用于 UI 警示，不再强制锁死买单)
        # 真正的风险控制由 RegimeFilter 的 TREND 模式 和 capital_below_ma_long 承载
        ma_series = df["close"].rolling(self.config.ma_long_period).mean()
        ma_now    = float(ma_series.iloc[-1])
        ma_prev   = float(ma_series.iloc[-(self.config.ma200_slope_lookback + 1)])
        ma_slope  = (ma_now - ma_prev) / ma_prev if ma_prev != 0 else 0
        
        # 仅作为状态标识
        downtrend_protection_triggered = (not is_bull) and (ma_slope < self.config.ma200_slope_threshold)

        # 6. 资金分配
        #   买方：USDT 部分  (active_capital * usdt_ratio)
        #   卖方：BTC 等值部分（剩余按当前价格估算为 BTC）
        current_usdt_ratio = self.config.usdt_ratio
        if self.config.dynamic_usdt_ratio_enabled and (not is_bull):
            current_usdt_ratio = self.config.bear_market_usdt_ratio
            logger.debug("空头背景：调低买单权重 | usdt_ratio: %.2f -> %.2f", self.config.usdt_ratio, current_usdt_ratio)

        usdt_for_buys = active_capital * current_usdt_ratio
        btc_for_sells = (active_capital * (1 - current_usdt_ratio)) / latest_close

        # 7. 生成挂单（含 qty 计算）
        buy_orders:  list = []
        sell_orders: list = []

        # 恢复逻辑：超买保护时锁买单；而下行趋势下仅减仓（通过 capital_below_ma_long），不再锁死。
        buy_grid_enabled = not overbought_triggered

        if buy_grid_enabled:
            price_step = (ma_mid - lower) / self.config.grid_levels
            n_buy = self.config.grid_levels
            usdt_per_level = usdt_for_buys / n_buy if n_buy > 0 else 0

            for i in range(1, n_buy + 1):
                price = ma_mid - i * price_step
                if price <= 0:
                    break
                qty = usdt_per_level / price
                # 过滤不满足最小要求的层
                if usdt_per_level < self.config.min_order_value or qty < self.config.min_order_qty:
                    qty = 0.0
                buy_orders.append({"price": round(price, 2), "qty": round(qty, 6)})

        price_step_up = (upper - ma_mid) / self.config.grid_levels
        n_sell = self.config.grid_levels
        btc_per_level = btc_for_sells / n_sell if n_sell > 0 else 0

        for i in range(1, n_sell + 1):
            price = ma_mid + i * price_step_up
            notional = btc_per_level * price
            qty = btc_per_level
            if notional < self.config.min_order_value or qty < self.config.min_order_qty:
                qty = 0.0
            sell_orders.append({"price": round(price, 2), "qty": round(qty, 6)})

        result = {
            "mid_price":            round(ma_mid,       2),
            "upper_boundary":       round(upper,        2),
            "lower_boundary":       round(lower,        2),
            "capital_utilization":  round(capital_util, 3),
            "active_capital":       round(active_capital, 2),
            "buy_orders":           buy_orders,
            "sell_orders":          sell_orders,
            "overbought_triggered": overbought_triggered,
            "downtrend_protection": downtrend_protection_triggered,
            "bias_pct":             round(bias * 100,   2),
            "is_bull_market":       is_bull,
            "buy_grid_enabled":     buy_grid_enabled,
            "sell_grid_enabled":    True,   # 超买保护只停买，不停卖
        }

        logger.debug("网格计算完成: mid=%.2f util=%.1f%% buy=%d层 sell=%d层",
                     ma_mid, capital_util * 100, len(buy_orders), len(sell_orders))
        return result

    # ------------------------------------------------------------------
    # 私有指标计算
    # ------------------------------------------------------------------

    def _calc_atr(self, df: pd.DataFrame, period: int) -> float:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def _calc_bb_upper(self, df: pd.DataFrame, period: int, std_mult: float) -> float:
        close = df["close"]
        mid   = close.rolling(period).mean()
        std   = close.rolling(period).std(ddof=0)
        return float((mid + std_mult * std).iloc[-1])


