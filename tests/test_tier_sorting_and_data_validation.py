"""
测试 RegimeFilter tiers 排序和 GridEngine 数据校验
Ticket 要求：
1. RegimeFilter tiers 应自动排序并校验
2. GridEngine 应在数据不足或 NaN/Inf 时抛出 ValueError
"""

import pytest
import pandas as pd
import numpy as np
from src.regime_filter import RegimeFilter, RegimeFilterConfig
from src.grid_engine import GridEngine, GridEngineConfig
from src.strategy_config import TierConfig


class TestRegimeFilterTiersSorting:
    """测试 RegimeFilter tiers 的排序和校验逻辑"""

    def test_tiers_auto_sort_on_init(self):
        """测试：乱序输入的 tiers 会被自动排序"""
        # 故意乱序
        tiers = [
            TierConfig(hurst_max=0.55, position_ratio=0.5, accumulate_spot=False),
            TierConfig(hurst_max=0.35, position_ratio=1.0, accumulate_spot=True),
            TierConfig(hurst_max=0.50, position_ratio=1.0, accumulate_spot=False),
        ]
        
        rf = RegimeFilter(tiers=tiers)
        
        # 排序后应该是 0.35, 0.50, 0.55
        assert rf.tiers[0].hurst_max == 0.35
        assert rf.tiers[1].hurst_max == 0.50
        assert rf.tiers[2].hurst_max == 0.55

    def test_tier_selection_with_sorted_tiers(self):
        """测试：排序后的 tiers 选择逻辑稳定"""
        tiers = [
            TierConfig(hurst_max=0.35, position_ratio=1.0, accumulate_spot=True),
            TierConfig(hurst_max=0.50, position_ratio=1.0, accumulate_spot=False),
            TierConfig(hurst_max=0.55, position_ratio=0.5, accumulate_spot=False),
        ]
        
        rf = RegimeFilter(tiers=tiers)
        
        # 构造一段震荡行情（Hurst < 0.5）
        df = self._create_range_market(length=200)
        result = rf.get_market_regime(df)
        
        # Hurst 通常 < 0.5，应该选中前两层之一
        # 验证 position_ratio 不为 0（即有有效选中）
        assert result.position_ratio > 0

    def test_tier_exceeding_all_tiers_returns_zero_position(self):
        """测试：当 Hurst 超出所有层级上限时，position_ratio 为 0"""
        tiers = [
            TierConfig(hurst_max=0.35, position_ratio=1.0, accumulate_spot=True),
            TierConfig(hurst_max=0.50, position_ratio=1.0, accumulate_spot=False),
        ]
        
        rf = RegimeFilter(tiers=tiers)
        
        # 构造一段趋势行情（Hurst > 0.5）
        df = self._create_trend_market(length=200)
        result = rf.get_market_regime(df)
        
        # 如果 Hurst > 0.5，超过所有层级，应该 position_ratio = 0
        # 注意：TREND/FUSE 状态也会强制 position_ratio = 0
        # 这里主要验证"超出层级"的逻辑
        if result.hurst_value > 0.50:
            assert result.position_ratio == 0.0
            assert result.accumulate_spot is False

    def test_tier_validation_duplicate_hurst_max(self):
        """测试：重复的 hurst_max 应该通过排序后校验失败"""
        tiers = [
            TierConfig(hurst_max=0.50, position_ratio=1.0, accumulate_spot=False),
            TierConfig(hurst_max=0.50, position_ratio=0.5, accumulate_spot=False),
        ]
        
        with pytest.raises(ValueError, match="必须严格大于前一层级"):
            RegimeFilter(tiers=tiers)

    def test_tier_validation_position_ratio_out_of_range(self):
        """测试：position_ratio 超出 [0, 1] 应该抛出 ValueError"""
        tiers = [
            TierConfig(hurst_max=0.35, position_ratio=1.5, accumulate_spot=False),
        ]
        
        with pytest.raises(ValueError, match=r"必须在 \[0, 1\] 范围内"):
            RegimeFilter(tiers=tiers)

    def test_tier_validation_negative_position_ratio(self):
        """测试：负的 position_ratio 应该抛出 ValueError"""
        tiers = [
            TierConfig(hurst_max=0.35, position_ratio=-0.1, accumulate_spot=False),
        ]
        
        with pytest.raises(ValueError, match=r"必须在 \[0, 1\] 范围内"):
            RegimeFilter(tiers=tiers)

    # 辅助方法
    def _create_range_market(self, length: int) -> pd.DataFrame:
        """生成均值回归行情（Ornstein-Uhlenbeck Process）"""
        price = np.zeros(length)
        price[0] = 100.0
        theta = 0.15
        mean_price = 100.0
        sigma = 1.0
        
        np.random.seed(42)
        
        for i in range(1, length):
            shock = np.random.normal(0, sigma)
            price[i] = price[i-1] + theta * (mean_price - price[i-1]) + shock
        
        return pd.DataFrame({
            'open': price,
            'high': price + 1.0,
            'low': price - 1.0,
            'close': price + 0.1,
            'volume': 1000.0 + np.abs(np.random.normal(0, 100, length))
        }, index=pd.date_range(start='2024-01-01', periods=length, freq='1h'))

    def _create_trend_market(self, length: int) -> pd.DataFrame:
        """生成单边上涨行情"""
        price = np.linspace(100, 200, length)
        
        return pd.DataFrame({
            'open': price,
            'high': price + 1,
            'low': price - 1,
            'close': price,
            'volume': 1000.0,
        }, index=pd.date_range(start='2024-01-01', periods=length, freq='1h'))


class TestGridEngineDataValidation:
    """测试 GridEngine 的数据不足和 NaN/Inf 校验逻辑"""

    def test_insufficient_data_raises_valueerror(self):
        """测试：数据不足时抛出 ValueError"""
        ge = GridEngine()
        
        # 创建不足的数据（少于 ma_long_period + ma200_slope_lookback）
        df = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [101.0] * 50,
            'low': [99.0] * 50,
            'close': [100.0] * 50,
            'volume': [1000.0] * 50,
        }, index=pd.date_range(start='2024-01-01', periods=50, freq='1h'))
        
        with pytest.raises(ValueError, match="数据不足"):
            ge.calculate_grid(df, total_balance=10000.0, position_ratio=1.0)

    def test_nan_in_close_raises_valueerror(self):
        """测试：close 价格为 NaN 时抛出 ValueError"""
        ge = GridEngine()
        
        # 创建足够的数据，但最后一个 close 为 NaN
        df = pd.DataFrame({
            'open': [100.0] * 300,
            'high': [101.0] * 300,
            'low': [99.0] * 300,
            'close': [100.0] * 299 + [np.nan],
            'volume': [1000.0] * 300,
        }, index=pd.date_range(start='2024-01-01', periods=300, freq='1h'))
        
        with pytest.raises(ValueError, match="数据不可用"):
            ge.calculate_grid(df, total_balance=10000.0, position_ratio=1.0)

    def test_ma_nan_raises_valueerror(self):
        """测试：MA 计算结果为 NaN 时抛出 ValueError"""
        ge = GridEngine()
        
        # 创建数据，但中间有 NaN 导致 rolling MA 为 NaN
        close_values = [100.0] * 150 + [np.nan] * 50 + [100.0] * 100
        df = pd.DataFrame({
            'open': close_values,
            'high': [v + 1 if pd.notna(v) else np.nan for v in close_values],
            'low': [v - 1 if pd.notna(v) else np.nan for v in close_values],
            'close': close_values,
            'volume': [1000.0] * 300,
        }, index=pd.date_range(start='2024-01-01', periods=300, freq='1h'))
        
        with pytest.raises(ValueError, match="指标不可用.*MA"):
            ge.calculate_grid(df, total_balance=10000.0, position_ratio=1.0)

    def test_sufficient_data_returns_valid_grid(self):
        """测试：足够的数据时，calculate_grid 返回有效结果"""
        ge = GridEngine()
        
        # 创建足够且有效的数据
        length = 300
        price = 100.0 + np.random.randn(length) * 2  # 围绕 100 波动
        df = pd.DataFrame({
            'open': price,
            'high': price + 1.0,
            'low': price - 1.0,
            'close': price,
            'volume': 1000.0 + np.abs(np.random.randn(length) * 100),
        }, index=pd.date_range(start='2024-01-01', periods=length, freq='1h'))
        
        result = ge.calculate_grid(df, total_balance=10000.0, position_ratio=1.0)
        
        # 验证结果结构完整
        assert 'mid_price' in result
        assert 'upper_boundary' in result
        assert 'lower_boundary' in result
        assert 'buy_orders' in result
        assert 'sell_orders' in result
        
        # 验证数值有效
        assert np.isfinite(result['mid_price'])
        assert np.isfinite(result['upper_boundary'])
        assert np.isfinite(result['lower_boundary'])
        assert isinstance(result['buy_orders'], list)
        assert isinstance(result['sell_orders'], list)

    def test_partial_nan_data_raises_valueerror(self):
        """测试：部分 NaN 数据导致 rolling 结果为 NaN 时抛出 ValueError"""
        ge = GridEngine()
        
        # 创建数据，前半部分有效，后半部分有几个 NaN
        close_values = [100.0] * 200 + [np.nan, 100.0, 100.0] + [100.0] * 97
        df = pd.DataFrame({
            'open': close_values,
            'high': [v + 1 if pd.notna(v) else np.nan for v in close_values],
            'low': [v - 1 if pd.notna(v) else np.nan for v in close_values],
            'close': close_values,
            'volume': [1000.0] * 300,
        }, index=pd.date_range(start='2024-01-01', periods=300, freq='1h'))
        
        # 如果 rolling window 包含 NaN，结果会是 NaN
        # 这应该被 isfinite 检查捕获
        with pytest.raises(ValueError, match="指标不可用"):
            ge.calculate_grid(df, total_balance=10000.0, position_ratio=1.0)
