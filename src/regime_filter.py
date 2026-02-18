"""
src/regime_filter.py - 市场状态过滤器 (Regime Filter)

通过多个技术指标综合判断当前市场处于"横盘震荡"还是"单边趋势"状态，
从而决定网格策略是否应当开启。

判断逻辑（投票机制）：
    - ADX (Average Directional Index): ADX 越高代表趋势越强
    - ATR 比例 (Short ATR / Long ATR): 比值突然放大代表波动率激增（趋势信号）
    - Bollinger Band Width: 带宽越宽代表波动越大，可能进入趋势
    - Hurst Exponent: < 0.5 均值回归（震荡），> 0.5 趋势性

每个指标独立投票，超过半数判断为趋势时，输出 TREND；否则输出 RANGE。
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from src.logger import setup_logger

logger = setup_logger("regime_filter")


class MarketRegime(str, Enum):
    """市场状态枚举。"""
    RANGE = "RANGE"  # 横盘震荡，适合网格
    TREND = "TREND"  # 单边趋势，暂停新开网格
    FUSE  = "FUSE"   # 熔断状态，紧急停止所有操作


class RegimeFilterConfig(BaseModel):
    """
    RegimeFilter 的参数配置（Pydantic 模型）。
    实例化时自动校验所有参数，非法值立即抛出 ValidationError。
    """

    # --- ADX ---
    adx_period: int = Field(default=14, ge=2, description="ADX 计算周期")
    adx_trend_threshold: float = Field(default=25.0, gt=0, le=100, description="ADX 超过此值视为非震荡")
    adx_fuse_threshold: float = Field(default=30.0, gt=0, le=100, description="ADX 超过此值需联合熔断")

    # --- ATR 比例 ---
    atr_short_period: int = Field(default=7, ge=2, description="短期 ATR 周期")
    atr_long_period: int = Field(default=28, ge=2, description="长期 ATR 周期")
    atr_ratio_threshold: float = Field(default=1.5, gt=0, description="短/长 ATR 比值超过此值视为趋势")

    # --- Bollinger Band Width ---
    bb_period: int = Field(default=20, ge=2, description="布林带计算周期")
    bb_std: float = Field(default=2.0, gt=0, description="布林带标准差倍数")
    bb_width_threshold: float = Field(default=0.1, gt=0, description="相对带宽超过此值视为趋势")

    # --- Hurst Exponent ---
    hurst_period: int = Field(default=100, ge=20, description="Hurst 指数计算所需的最少数据点")
    hurst_threshold: float = Field(default=0.5, gt=0, lt=1, description="Hurst 低于此值代表适合网格")
    hurst_fuse_threshold: float = Field(default=0.6, gt=0, lt=1, description="Hurst 超过此值需联合熔断")
    hurst_extreme_low: float = Field(default=0.35, gt=0, lt=1, description="Hurst 低于此值触发利润转现货")

    # --- 投票机制 ---
    trend_vote_threshold: int = Field(default=2, ge=1, le=4, description="判断为趋势所需的最少投票数")

    @model_validator(mode="after")
    def check_cross_field_constraints(self) -> RegimeFilterConfig:
        """
        跨字段约束校验。所有指标的周期参数必须满足以下逻辑一致性：

        1. atr_short_period < atr_long_period
           短期 ATR 必须严格小于长期 ATR，否则"比值"失去意义。

        2. hurst_period >= atr_long_period
           Hurst 的数据窗口必须能覆盖长期 ATR 的计算需求，
           否则在实际运行时会因数据不足导致 Hurst 计算退化。

        3. hurst_period >= bb_period
           同上，Hurst 窗口必须能覆盖布林带的计算需求。

        4. hurst_period >= adx_period * 2
           ADX 需要 2 倍周期的数据预热（先算 DI 再算 ADX），
           Hurst 窗口必须足够大以确保 ADX 已充分收敛。
        """
        errors = []

        if self.atr_short_period >= self.atr_long_period:
            errors.append(
                f"atr_short_period ({self.atr_short_period}) 必须严格小于 "
                f"atr_long_period ({self.atr_long_period})"
            )

        if self.hurst_period < self.atr_long_period:
            errors.append(
                f"hurst_period ({self.hurst_period}) 必须 >= "
                f"atr_long_period ({self.atr_long_period})，"
                f"否则数据窗口不足以支撑长期 ATR 的计算"
            )

        if self.hurst_period < self.bb_period:
            errors.append(
                f"hurst_period ({self.hurst_period}) 必须 >= "
                f"bb_period ({self.bb_period})，"
                f"否则数据窗口不足以支撑布林带的计算"
            )

        adx_warmup = self.adx_period * 2
        if self.hurst_period < adx_warmup:
            errors.append(
                f"hurst_period ({self.hurst_period}) 必须 >= "
                f"adx_period * 2 ({adx_warmup})，"
                f"否则数据窗口不足以支撑 ADX 的充分收敛"
            )

        if errors:
            raise ValueError("参数逻辑约束冲突:\n" + "\n".join(f"  - {e}" for e in errors))

        return self



@dataclass
class RegimeResult:
    """单次 Regime 判断的详细结果，便于调试和日志记录。"""
    regime: MarketRegime
    position_ratio: float   # 分级仓位比例 (0.0 to 1.0)
    accumulate_spot: bool   # 是否开启利润转现货 BTC
    fuse_triggered: bool    # 是否由于 Hurst/ADX 触发了熔断
    
    adx_value: float
    adx_vote: bool          # True = 趋势
    atr_ratio: float
    atr_vote: bool
    bb_width: float
    bb_vote: bool
    hurst_value: float
    hurst_vote: bool
    trend_votes: int        # 投票为趋势的指标数量
    total_votes: int        # 参与投票的指标总数

    def __str__(self) -> str:
        return (
            f"Regime={self.regime.value} | "
            f"Pos={self.position_ratio:.0%} | "
            f"SpotAccum={'ON' if self.accumulate_spot else 'OFF'} | "
            f"ADX={self.adx_value:.2f}({'T' if self.adx_vote else 'R'}) | "
            f"ATR_ratio={self.atr_ratio:.3f}({'T' if self.atr_vote else 'R'}) | "
            f"BB_width={self.bb_width:.4f}({'T' if self.bb_vote else 'R'}) | "
            f"Hurst={self.hurst_value:.3f}({'T' if self.hurst_vote else 'R'}) | "
            f"Votes={self.trend_votes}/{self.total_votes}"
        )


class RegimeFilter:
    """
    市场状态过滤器。

    使用多指标投票机制判断当前市场状态，支持对单个时间点或整个 DataFrame 进行批量判断。

    Example:
        >>> from src.regime_filter import RegimeFilter, RegimeFilterConfig
        >>> rf = RegimeFilter()
        >>> result = rf.get_market_regime(df)
        >>> print(result.regime)  # MarketRegime.RANGE 或 MarketRegime.TREND
    """

    def __init__(self, config: Optional[RegimeFilterConfig] = None) -> None:
        self.config = config or RegimeFilterConfig()
        logger.info("RegimeFilter 初始化完成 | 配置: %s", self.config)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_market_regime(self, df: pd.DataFrame) -> RegimeResult:
        """
        对传入的 OHLCV DataFrame 的最新时间点进行市场状态判断。

        Args:
            df: 包含 [open, high, low, close, volume] 列的 DataFrame，
                行数需满足各指标的最小计算周期。

        Returns:
            RegimeResult 对象，包含最终状态和各指标的详细数值。
        """
        min_required = max(
            self.config.adx_period * 2,
            self.config.atr_long_period * 2,
            self.config.bb_period,
            self.config.hurst_period,
        )
        if len(df) < min_required:
            raise ValueError(
                f"数据行数不足：需要至少 {min_required} 行，当前只有 {len(df)} 行。"
            )

        # 1. 计算指标数值
        adx_val = self._calc_adx(df)
        atr_ratio = self._calc_atr_ratio(df)
        bb_width = self._calc_bb_width(df)
        hurst_val = self._calc_hurst(df)

        # 2. 投票判断 (传统的四指标投票)
        adx_vote = adx_val > self.config.adx_trend_threshold
        atr_vote = atr_ratio > self.config.atr_ratio_threshold
        bb_vote = bb_width > self.config.bb_width_threshold
        hurst_vote = hurst_val > self.config.hurst_threshold
        trend_votes = sum([adx_vote, atr_vote, bb_vote, hurst_vote])

        # 3. v2.0 逻辑：核心开关与熔断
        fuse_triggered = (hurst_val > self.config.hurst_fuse_threshold and adx_val > self.config.adx_fuse_threshold)
        
        # 默认基于投票
        regime = (
            MarketRegime.TREND
            if trend_votes >= self.config.trend_vote_threshold
            else MarketRegime.RANGE
        )

        # 震荡增强：必须满足 Hurst < 0.5 且 ADX < 25
        if regime == MarketRegime.RANGE:
            if not (hurst_val < 0.5 and adx_val < 25):
                 # 如果投票是 RANGE 但未能满足严苛条件，退化为 TREND/WAIT
                 regime = MarketRegime.TREND 

        # 如果触发熔断，状态强制为 FUSE
        if fuse_triggered:
            regime = MarketRegime.FUSE

        # 4. 分级仓位响应 (Tiered Response based on Hurst)
        position_ratio = 1.0
        accumulate_spot = False

        if hurst_val < self.config.hurst_extreme_low:
            # 极低 Hurst (< 0.35)：反弹/变盘前兆，100% 仓位 + 利润转现货
            position_ratio = 1.0
            accumulate_spot = True
        elif hurst_val < 0.4:
            position_ratio = 1.0
        elif hurst_val < 0.5:
            position_ratio = 1.0
        elif hurst_val < 0.55:
            # 接近 0.5 随机游走，50% 仓位
            position_ratio = 0.5
        else:
            # Hurst 较高或趋势性，0 仓位
            position_ratio = 0.0
            if regime == MarketRegime.RANGE:
                 regime = MarketRegime.TREND # 修正状态

        # 强制修正：如果熔断或趋势判断明确，仓位清零（除非是极低 Hurst，但通常极低 Hurst 不会触发熔断）
        if regime in [MarketRegime.TREND, MarketRegime.FUSE]:
            position_ratio = 0.0

        result = RegimeResult(
            regime=regime,
            position_ratio=position_ratio,
            accumulate_spot=accumulate_spot,
            fuse_triggered=fuse_triggered,
            adx_value=adx_val,
            adx_vote=adx_vote,
            atr_ratio=atr_ratio,
            atr_vote=atr_vote,
            bb_width=bb_width,
            bb_vote=bb_vote,
            hurst_value=hurst_val,
            hurst_vote=hurst_vote,
            trend_votes=trend_votes,
            total_votes=4,
        )
        logger.info("Regime 判断结果: %s", result)
        return result

    def scan_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        对整个 DataFrame 进行逐行滚动判断，返回每个时间点的市场状态序列。
        用于回测时标注历史上哪些时段适合开启网格。

        Args:
            df: 完整的 OHLCV DataFrame。

        Returns:
            pd.Series，值为 "RANGE" 或 "TREND"，索引与 df 对齐。
            前 min_required 行因数据不足，填充为 NaN。
        """
        min_required = max(
            self.config.adx_period * 2,
            self.config.atr_long_period * 2,
            self.config.bb_period,
            self.config.hurst_period,
        )

        regimes = pd.Series(index=df.index, dtype=object)
        for i in range(min_required, len(df) + 1):
            window = df.iloc[:i]
            try:
                result = self.get_market_regime(window)
                regimes.iloc[i - 1] = result.regime.value
            except ValueError:
                pass

        logger.info(
            "历史 Regime 扫描完成 | 总计 %d 个时间点 | RANGE: %d | TREND: %d",
            regimes.notna().sum(),
            (regimes == "RANGE").sum(),
            (regimes == "TREND").sum(),
        )
        return regimes

    # ------------------------------------------------------------------
    # 私有指标计算方法
    # ------------------------------------------------------------------

    def _calc_adx(self, df: pd.DataFrame) -> float:
        """
        计算 ADX（平均趋向指数）。

        ADX 衡量趋势的强度（而非方向）：
            ADX < 20: 无趋势 / 震荡
            20-25:    弱趋势
            25-50:    强趋势
            > 50:     极强趋势（罕见）

        Returns:
            最新时间点的 ADX 值（0-100）。
        """
        n = self.config.adx_period
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        length = len(close)

        # True Range
        tr = np.zeros(length)
        plus_dm = np.zeros(length)
        minus_dm = np.zeros(length)

        for i in range(1, length):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

            up = high[i] - high[i - 1]
            dn = low[i - 1] - low[i]
            plus_dm[i] = up if (up > dn and up > 0) else 0.0
            minus_dm[i] = dn if (dn > up and dn > 0) else 0.0

        # Wilder's smoothing（初始种子用前 n 个元素的均值，之后用 Wilder 递推公式）
        def wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
            result = np.zeros(length)
            # 初始种子：第 1 到 period 个元素的简单均值（跳过 index 0 因为它是 0）
            result[period] = np.mean(arr[1:period + 1])
            for i in range(period + 1, length):
                result[i] = result[i - 1] * (1 - 1.0 / period) + arr[i] * (1.0 / period)
            return result

        atr_s = wilder_smooth(tr, n)
        plus_s = wilder_smooth(plus_dm, n)
        minus_s = wilder_smooth(minus_dm, n)

        # DI 计算
        with np.errstate(divide='ignore', invalid='ignore'):
            plus_di = np.where(atr_s > 0, 100 * plus_s / atr_s, 0.0)
            minus_di = np.where(atr_s > 0, 100 * minus_s / atr_s, 0.0)
            di_sum = plus_di + minus_di
            dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0.0)

        # ADX = Wilder 平滑的 DX（从 2*n 位置开始才有意义）
        adx = wilder_smooth(dx, n)
        return float(adx[-1])


    def _calc_atr_ratio(self, df: pd.DataFrame) -> float:
        """
        计算短期 ATR / 长期 ATR 的比值。

        比值 > 1.5 代表近期波动率相对历史显著放大，通常是趋势启动的信号。

        Returns:
            短期 ATR / 长期 ATR 的比值。
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr_short = tr.ewm(span=self.config.atr_short_period, adjust=False).mean().iloc[-1]
        atr_long = tr.ewm(span=self.config.atr_long_period, adjust=False).mean().iloc[-1]

        if atr_long == 0:
            return 1.0  # 防止除以零，返回中性值
        return float(atr_short / atr_long)

    def _calc_bb_width(self, df: pd.DataFrame) -> float:
        """
        计算布林带相对带宽 (Bollinger Band Width)。

        带宽 = (上轨 - 下轨) / 中轨
        带宽越大代表波动越剧烈，可能处于趋势行情。
        带宽极小（Squeeze）代表即将突破，但当下仍是震荡。

        Returns:
            最新时间点的相对带宽值（无量纲）。
        """
        close = df["close"]
        n = self.config.bb_period
        std_mult = self.config.bb_std

        mid = close.rolling(n).mean()
        std = close.rolling(n).std(ddof=0)

        upper = mid + std_mult * std
        lower = mid - std_mult * std

        width = (upper - lower) / mid.replace(0, np.nan)
        return float(width.iloc[-1])

    def _calc_hurst(self, df: pd.DataFrame) -> float:
        """
        计算 Hurst 指数（R/S 分析法）。

        Hurst 指数衡量时间序列的长期记忆性：
            H < 0.5: 均值回归（反持续性），适合网格策略
            H ≈ 0.5: 随机游走（无记忆性）
            H > 0.5: 趋势持续性，不适合网格策略

        Returns:
            Hurst 指数（0-1 之间）。
        """
        close = df["close"].values[-self.config.hurst_period:]
        n = len(close)

        if n < 20:
            return 0.5  # 数据不足时返回中性值

        # 使用对数收益率（而非价格本身），更符合金融时序的统计特性
        log_returns = np.diff(np.log(close))
        m = len(log_returns)

        # 选取多个不同的 lag 尺度进行 R/S 分析
        max_lag = m // 2
        lags = np.unique(np.logspace(1, np.log10(max_lag), num=20).astype(int))
        lags = lags[lags >= 8]  # 过滤掉太短的 lag

        rs_values = []
        for lag in lags:
            # 对整个序列按 lag 长度分段，计算每段的 R/S
            rs_per_seg = []
            for start in range(0, m - lag + 1, lag):
                seg = log_returns[start: start + lag]
                mean = np.mean(seg)
                deviation = np.cumsum(seg - mean)
                r = np.max(deviation) - np.min(deviation)
                s = np.std(seg, ddof=1)
                if s > 0:
                    rs_per_seg.append(r / s)

            if len(rs_per_seg) >= 2:
                rs_values.append((lag, np.mean(rs_per_seg)))

        if len(rs_values) < 4:
            return 0.5

        lags_arr = np.log([x[0] for x in rs_values])
        rs_arr = np.log([x[1] for x in rs_values])

        # 线性回归斜率即为 Hurst 指数
        hurst = float(np.polyfit(lags_arr, rs_arr, 1)[0])
        return float(np.clip(hurst, 0.0, 1.0))
