"""
src/grid_engine.py - 自适应网格逻辑引擎 (v2.0)

负责计算网格中轴线、上下边界、资金利用率以及生成具体的挂单列表。
支持以下核心特性：
    - MA200/120/60 资金分配逻辑
    - ATR 动态边界锁定
    - MA20 中轴漂移
    - 布林带 2.5σ + MA 偏离度超买保护
"""

from __future__ import annotations

from typing import List, Optional
import pandas as pd
from pydantic import BaseModel, Field

from src.logger import setup_logger

logger = setup_logger("grid_engine")


class GridEngineConfig(BaseModel):
    """自适应网格引擎参数。"""
    
    # --- 基准线 ---
    ma_long_period: int = Field(default=200, ge=10)
    ma_mid_period: int = Field(default=20, ge=5)
    
    # --- 资金利用率 ---
    capital_above_ma_long: float = Field(default=0.8, gt=0, le=1.0)
    capital_below_ma_long: float = Field(default=0.3, gt=0, le=1.0)
    
    # --- 动态区间 (ATR) ---
    atr_band_multiplier: float = Field(default=3.0, gt=0)
    atr_period: int = Field(default=14, ge=2)
    
    # --- 超买保护 ---
    overbought_bb_std: float = Field(default=2.5, gt=0)
    overbought_bb_period: int = Field(default=20, ge=2)
    bias_overbought_threshold: float = Field(default=0.05, gt=0)
    
    # --- 网格设置 ---
    grid_levels: int = Field(default=10, ge=2)
    min_grid_spacing_pct: float = Field(default=0.003, gt=0)
    max_grid_spacing_pct: float = Field(default=0.05, gt=0)


class GridEngine:
    """自适应网格计算引擎。"""

    def __init__(self, config: Optional[GridEngineConfig] = None) -> None:
        self.config = config or GridEngineConfig()
        logger.info("GridEngine 初始化完成 | 配置: %s", self.config)

    def calculate_grid(
        self, 
        df: pd.DataFrame, 
        total_balance: float,
        position_ratio: float = 1.0
    ) -> dict:
        """
        根据最新市场数据计算网格计划。
        
        Returns:
            dict 包含:
                - upper_boundary: 网格上限
                - lower_boundary: 网格下限
                - mid_price: 中轴价 (MA20)
                - capital_utilization: 资金利用率
                - buy_orders: [{price, qty}, ...]
                - sell_orders: [{price, qty}, ...]
                - overbought_triggered: bool
        """
        if len(df) < self.config.ma_long_period:
            raise ValueError(f"数据不足：需要至少 {self.config.ma_long_period} 根 K 线计算基准线")

        latest_close = df["close"].iloc[-1]
        
        # 1. 计算基准线 (MA200, MA20)
        ma_long = df["close"].rolling(self.config.ma_long_period).mean().iloc[-1]
        ma_mid = df["close"].rolling(self.config.ma_mid_period).mean().iloc[-1]
        
        # 2. 资金利用率分配
        capital_base = self.config.capital_above_ma_long if latest_close > ma_long else self.config.capital_below_ma_long
        active_capital = total_balance * capital_base * position_ratio
        
        # 3. 动态边界 (ATR)
        atr = self._calc_atr(df, self.config.atr_period)
        upper = latest_close + (self.config.atr_band_multiplier * atr)
        lower = latest_close - (self.config.atr_band_multiplier * atr)
        
        # 4. 超买保护 (BB 2.5σ + Bias)
        bias = (latest_close - ma_mid) / ma_mid
        bb_upper = self._calc_bb_upper(df, self.config.overbought_bb_period, self.config.overbought_bb_std)
        overbought_triggered = (latest_close >= bb_upper and bias > self.config.bias_overbought_threshold)
        
        # 5. 生成挂单
        buy_orders = []
        sell_orders = []

        if not overbought_triggered:
            # 买单：从中轴向下分布到下轨
            # 简单的等差数列分布示例 (实际策略可能需要等比或动态间距)
            price_step = (ma_mid - lower) / self.config.grid_levels
            for i in range(1, self.config.grid_levels + 1):
                price = ma_mid - i * price_step
                if price <= 0: break
                buy_orders.append({"price": round(price, 2), "qty": 0.0}) # qty 需结合资金计算

        # 卖单：从中轴向上分布到上轨
        price_step_up = (upper - ma_mid) / self.config.grid_levels
        for i in range(1, self.config.grid_levels + 1):
            price = ma_mid + i * price_step_up
            sell_orders.append({"price": round(price, 2), "qty": 0.0})

        buy_enabled = not overbought_triggered
        sell_enabled = True # 超买保护只停买，不停卖

        result = {
            "mid_price": round(ma_mid, 2),
            "upper_boundary": round(upper, 2),
            "lower_boundary": round(lower, 2),
            "capital_utilization": round(capital_base * position_ratio, 2),
            "buy_orders": buy_orders,
            "sell_orders": sell_orders,
            "overbought_triggered": overbought_triggered,
            "bias_pct": round(bias * 100, 2),
            "is_bull_market": latest_close > ma_long,
            "buy_grid_enabled": buy_enabled,
            "sell_grid_enabled": sell_enabled,
        }
        
        logger.info("网格计算完成: %s", result)
        return result

    def _calc_atr(self, df: pd.DataFrame, period: int) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def _calc_bb_upper(self, df: pd.DataFrame, period: int, std_mult: float) -> float:
        close = df["close"]
        mid = close.rolling(period).mean()
        std = close.rolling(period).std(ddof=0)
        return float((mid + std_mult * std).iloc[-1])
