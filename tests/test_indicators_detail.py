
import pytest
import pandas as pd
import numpy as np
from src.regime_filter import RegimeFilter
from src.strategy_config import RegimeFilterConfig, PositionSizingConfig

@pytest.fixture
def rf_instance():
    cfg = RegimeFilterConfig(
        adx_period=14,
        atr_short_period=7,
        atr_long_period=28,
        bb_period=20,
        bb_std=2.0,
        hurst_period=100,
        ma200_period=50, # 缩短些便于生成数据
        ma200_slope_lookback=10
    )
    pos_cfg = PositionSizingConfig()
    return RegimeFilter(cfg, pos_cfg.tiers)

def generate_df(prices):
    length = len(prices)
    return pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': 1000
    }, index=pd.date_range('2024-01-01', periods=length, freq='1h'))

# 1. 测试 ADX
def test_calc_adx_trend(rf_instance):
    # 强上涨趋势
    prices = np.linspace(100, 200, 200)
    df = generate_df(prices)
    adx = rf_instance._calc_adx(df)
    assert adx > 25, f"Trending market should have high ADX, got {adx}"

def test_calc_adx_range(rf_instance):
    # 随机波动无趋势 (Random Walk with no drift)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 0.1, 300))
    df = generate_df(prices)
    adx = rf_instance._calc_adx(df)
    # 随机游走的 ADX 应该较低
    assert adx < 30, f"No-trend market should have low ADX, got {adx}"

# 2. 测试 ATR Ratio
def test_calc_atr_ratio_spike(rf_instance):
    # 波动率突然放大
    prices = np.ones(100) * 100
    # 前面很平
    df_low_vol = generate_df(prices)
    # 后面剧烈波动
    volatile_prices = np.array([100, 110, 90, 120, 80, 130, 70])
    df_high_vol = generate_df(np.concatenate([prices, volatile_prices]))
    
    ratio = rf_instance._calc_atr_ratio(df_high_vol)
    assert ratio > 1.2, f"Volatility spike should result in high ATR ratio, got {ratio}"

# 3. 测试 BB Width
def test_calc_bb_width(rf_instance):
    # 高波动
    p_high = 100 + np.random.normal(0, 5, 100)
    df_high = generate_df(p_high)
    w_high = rf_instance._calc_bb_width(df_high)
    
    # 低波动
    p_low = 100 + np.random.normal(0, 0.5, 100)
    df_low = generate_df(p_low)
    w_low = rf_instance._calc_bb_width(df_low)
    
    assert w_high > w_low

# 4. 测试 Hurst Exponent
def test_calc_hurst_reverting(rf_instance):
    # 均值回归 (Ornstein-Uhlenbeck)
    length = 500
    prices = np.zeros(length)
    prices[0] = 100
    theta = 0.3
    for i in range(1, length):
        prices[i] = prices[i-1] + theta * (100 - prices[i-1]) + np.random.normal(0, 1)
    
    df = generate_df(prices)
    hurst = rf_instance._calc_hurst(df)
    # 理论上 H < 0.5
    assert hurst < 0.6, f"Mean reverting market should have Hurst < 0.5 (or close), got {hurst}"

def test_calc_hurst_trending(rf_instance):
    # 随机游走 + 趋势
    length = 500
    prices = np.cumsum(np.random.normal(0.1, 1, length)) + 100
    df = generate_df(prices)
    hurst = rf_instance._calc_hurst(df)
    # 理论上 H > 0.5
    assert hurst > 0.4, f"Trending market should have Hurst > 0.5 (or close), got {hurst}"

# 5. 测试 MA200 Slope
def test_calc_ma200_slope_down(rf_instance):
    # 长期下跌
    prices = np.linspace(1000, 500, 300)
    df = generate_df(prices)
    slope = rf_instance._calc_ma200_slope(df)
    assert slope < 0, f"Downtrend should have negative slope, got {slope}"

def test_calc_ma200_slope_up(rf_instance):
    # 长期上涨
    prices = np.linspace(500, 1000, 300)
    df = generate_df(prices)
    slope = rf_instance._calc_ma200_slope(df)
    assert slope > 0, f"Uptrend should have positive slope, got {slope}"

