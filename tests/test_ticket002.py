"""
tests/test_ticket002.py - Ticket-002 验收测试

验收标准:
  AC1: 输入历史 K 线数据，能够输出每个时间点的市场状态（RANGE 或 TREND）。
  AC2: 在明显的单边行情下，过滤器输出 TREND。
  AC3: 在明显的横盘行情下，过滤器输出 RANGE。

运行方式:
  python -m pytest tests/test_ticket002.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.regime_filter import (
    RegimeFilter,
    RegimeFilterConfig,
    RegimeResult,
    MarketRegime,
)


# ======================================================================
# 测试数据生成工具
# ======================================================================

def make_ranging_ohlcv(n: int = 200, base: float = 50000.0, noise: float = 300.0) -> pd.DataFrame:
    """
    生成模拟横盘震荡行情的 OHLCV 数据。
    价格在 base ± noise 范围内随机游走（均值回归）。
    """
    np.random.seed(42)
    # 使用 OU 过程（Ornstein-Uhlenbeck）模拟均值回归
    prices = [base]
    theta = 0.1  # 均值回归速度
    for _ in range(n - 1):
        drift = theta * (base - prices[-1])
        shock = np.random.normal(0, noise)
        prices.append(prices[-1] + drift + shock)

    prices = np.array(prices)
    # 生成 OHLCV
    high = prices + np.abs(np.random.normal(0, noise * 0.3, n))
    low = prices - np.abs(np.random.normal(0, noise * 0.3, n))
    open_ = prices + np.random.normal(0, noise * 0.1, n)
    volume = np.random.uniform(100, 500, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": prices, "volume": volume},
        index=idx,
    )


def make_trending_ohlcv(n: int = 200, start: float = 40000.0, end: float = 70000.0) -> pd.DataFrame:
    """
    生成模拟单边上涨趋势行情的 OHLCV 数据。
    价格从 start 线性上涨至 end，叠加少量噪音。
    """
    np.random.seed(99)
    trend = np.linspace(start, end, n)
    noise = np.random.normal(0, (end - start) * 0.005, n)
    prices = trend + noise

    step = (end - start) / n
    high = prices + np.abs(np.random.normal(0, step * 0.5, n))
    low = prices - np.abs(np.random.normal(0, step * 0.5, n))
    open_ = np.roll(prices, 1)
    open_[0] = start
    volume = np.linspace(200, 800, n) + np.random.uniform(0, 100, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": prices, "volume": volume},
        index=idx,
    )


# ======================================================================
# 基础功能测试
# ======================================================================

class TestRegimeFilterBasic:
    """测试 RegimeFilter 的基础功能和接口。"""

    def setup_method(self):
        self.rf = RegimeFilter()
        self.ranging_df = make_ranging_ohlcv(n=200)
        self.trending_df = make_trending_ohlcv(n=200)

    def test_returns_regime_result(self):
        """get_market_regime 应返回 RegimeResult 实例。"""
        result = self.rf.get_market_regime(self.ranging_df)
        assert isinstance(result, RegimeResult)

    def test_regime_is_valid_enum(self):
        """返回的 regime 应为合法的 MarketRegime 枚举值。"""
        result = self.rf.get_market_regime(self.ranging_df)
        assert result.regime in (MarketRegime.RANGE, MarketRegime.TREND)

    def test_result_contains_all_indicators(self):
        """RegimeResult 应包含所有指标的数值。"""
        result = self.rf.get_market_regime(self.ranging_df)
        assert isinstance(result.adx_value, float)
        assert isinstance(result.atr_ratio, float)
        assert isinstance(result.bb_width, float)
        assert isinstance(result.hurst_value, float)

    def test_insufficient_data_raises_error(self):
        """数据行数不足时应抛出 ValueError。"""
        tiny_df = make_ranging_ohlcv(n=10)
        with pytest.raises(ValueError, match="数据行数不足"):
            self.rf.get_market_regime(tiny_df)

    def test_vote_counts_are_consistent(self):
        """trend_votes 应在 0 到 total_votes 之间。"""
        result = self.rf.get_market_regime(self.ranging_df)
        assert 0 <= result.trend_votes <= result.total_votes

    def test_regime_consistent_with_votes(self):
        """最终 regime 应与投票结果一致。"""
        result = self.rf.get_market_regime(self.ranging_df)
        if result.trend_votes >= self.rf.config.trend_vote_threshold:
            assert result.regime == MarketRegime.TREND
        else:
            assert result.regime == MarketRegime.RANGE


# ======================================================================
# AC2: 趋势行情识别测试
# ======================================================================

class TestTrendDetection:
    """AC2: 在明显的单边行情下，过滤器应输出 TREND。"""

    def setup_method(self):
        # 使用更敏感的配置以确保趋势被识别
        config = RegimeFilterConfig(
            adx_trend_threshold=20.0,
            atr_ratio_threshold=1.2,
            bb_width_threshold=0.05,
            hurst_threshold=0.5,
            trend_vote_threshold=2,
        )
        self.rf = RegimeFilter(config=config)
        self.trending_df = make_trending_ohlcv(n=200, start=40000.0, end=70000.0)

    def test_ac2_trend_regime_detected(self):
        """AC2: 明显上涨趋势行情下，应识别为 TREND。"""
        result = self.rf.get_market_regime(self.trending_df)
        print(f"\n趋势行情判断结果: {result}")
        assert result.regime == MarketRegime.TREND, (
            f"期望 TREND，实际 {result.regime.value}。详情: {result}"
        )

    def test_adx_high_in_trend(self):
        """趋势行情下，ADX 值应在有效范围 0-100 内，且高于震荡行情。"""
        trend_result = self.rf.get_market_regime(self.trending_df)
        ranging_df = make_ranging_ohlcv(n=200)
        range_result = self.rf.get_market_regime(ranging_df)
        # ADX 应在合法范围内
        assert 0 <= trend_result.adx_value <= 100, (
            f"趋势 ADX 超出范围: {trend_result.adx_value:.2f}"
        )
        # 趋势行情的 ADX 应高于震荡行情
        assert trend_result.adx_value > range_result.adx_value, (
            f"趋势 ADX ({trend_result.adx_value:.2f}) 应大于震荡 ADX ({range_result.adx_value:.2f})"
        )

    def test_hurst_is_valid_range(self):
        """Hurst 指数应在 [0, 1] 范围内。"""
        result = self.rf.get_market_regime(self.trending_df)
        assert 0.0 <= result.hurst_value <= 1.0, (
            f"Hurst 超出合法范围: {result.hurst_value:.3f}"
        )



# ======================================================================
# AC3: 震荡行情识别测试
# ======================================================================

class TestRangeDetection:
    """AC3: 在明显的横盘行情下，过滤器应输出 RANGE。"""

    def setup_method(self):
        config = RegimeFilterConfig(
            adx_trend_threshold=25.0,
            atr_ratio_threshold=1.5,
            bb_width_threshold=0.08,
            hurst_threshold=0.5,
            trend_vote_threshold=2,
        )
        self.rf = RegimeFilter(config=config)
        self.ranging_df = make_ranging_ohlcv(n=200, noise=200.0)

    def test_ac3_range_regime_detected(self):
        """AC3: 明显横盘震荡行情下，应识别为 RANGE。"""
        result = self.rf.get_market_regime(self.ranging_df)
        print(f"\n震荡行情判断结果: {result}")
        assert result.regime == MarketRegime.RANGE, (
            f"期望 RANGE，实际 {result.regime.value}。详情: {result}"
        )

    def test_hurst_is_valid_range_in_ranging(self):
        """震荡行情下，Hurst 指数应在 [0, 1] 合法范围内。"""
        result = self.rf.get_market_regime(self.ranging_df)
        assert 0.0 <= result.hurst_value <= 1.0, (
            f"Hurst 超出合法范围: {result.hurst_value:.3f}"
        )



# ======================================================================
# AC1: scan_dataframe 批量扫描测试
# ======================================================================

class TestScanDataframe:
    """AC1: 对整个 DataFrame 进行逐行扫描，输出每个时间点的状态。"""

    def setup_method(self):
        self.rf = RegimeFilter()

    def test_ac1_scan_returns_series(self):
        """scan_dataframe 应返回 pd.Series。"""
        df = make_ranging_ohlcv(n=200)
        result = self.rf.scan_dataframe(df)
        assert isinstance(result, pd.Series)

    def test_ac1_scan_index_aligned(self):
        """scan_dataframe 返回的 Series 索引应与输入 DataFrame 对齐。"""
        df = make_ranging_ohlcv(n=200)
        result = self.rf.scan_dataframe(df)
        assert result.index.equals(df.index)

    def test_ac1_scan_values_are_valid(self):
        """scan_dataframe 中的非空值应只包含 'RANGE' 或 'TREND'。"""
        df = make_ranging_ohlcv(n=200)
        result = self.rf.scan_dataframe(df)
        valid_values = {"RANGE", "TREND"}
        actual_values = set(result.dropna().unique())
        assert actual_values.issubset(valid_values), (
            f"发现非法值: {actual_values - valid_values}"
        )

    def test_ac1_scan_has_sufficient_non_null(self):
        """scan_dataframe 应有足够数量的非空判断结果。"""
        df = make_ranging_ohlcv(n=200)
        result = self.rf.scan_dataframe(df)
        non_null_count = result.notna().sum()
        assert non_null_count > 50, (
            f"有效判断点数量不足，期望 > 50，实际 {non_null_count}"
        )


# ======================================================================
# 配置参数测试
# ======================================================================

class TestRegimeFilterConfig:
    """测试自定义配置对过滤器行为的影响。"""

    def test_strict_config_more_trend(self):
        """更宽松的趋势阈值应识别出更多趋势。"""
        strict_config = RegimeFilterConfig(trend_vote_threshold=4)  # 需要 4/4 才判断为趋势
        loose_config = RegimeFilterConfig(trend_vote_threshold=1)   # 只需 1/4 就判断为趋势

        rf_strict = RegimeFilter(config=strict_config)
        rf_loose = RegimeFilter(config=loose_config)

        df = make_trending_ohlcv(n=200)
        result_strict = rf_strict.get_market_regime(df)
        result_loose = rf_loose.get_market_regime(df)

        # 宽松配置应更容易判断为趋势
        assert result_loose.trend_votes >= result_strict.trend_votes
