
import pytest
import pandas as pd
import numpy as np
from src.regime_filter import RegimeFilter, MarketRegime, RegimeResult
from src.grid_engine import GridEngine, GridEngineConfig
from src.strategy_config import RegimeFilterConfig, PositionSizingConfig
from src.backtest import BacktestEngine
from src.feed import BacktestFeed
from src.feed.base import TickData

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def mock_ohlcv_range():
    """生成一段均值回归行情 (Ornstein-Uhlenbeck Process)"""
    length = 500
    price = np.zeros(length)
    price[0] = 100.0
    theta = 0.15 # 回归力度 (越大回归越快)
    mean_price = 100.0
    sigma = 1.0
    
    np.random.seed(42) # 固定随机种子以便复现
    
    for i in range(1, length):
        shock = np.random.normal(0, sigma)
        price[i] = price[i-1] + theta * (mean_price - price[i-1]) + shock
    
    df = pd.DataFrame({
        'open': price,
        'high': price + 1.0, # 简单扩展 High/Low
        'low': price - 1.0,
        'close': price + 0.1, 
        'volume': 1000.0 + np.abs(np.random.normal(0, 100, length))
    }, index=pd.date_range(start='2024-01-01', periods=length, freq='1h'))
    return df

@pytest.fixture
def mock_ohlcv_trend():
    """生成一段单边上涨行情 (Linear Trend)"""
    length = 500
    price = np.linspace(100, 200, length)
    
    df = pd.DataFrame({
        'open': price,
        'high': price + 1,
        'low': price - 1,
        'close': price + 0.5,
        'volume': 1000
    }, index=pd.date_range(start='2024-01-01', periods=length, freq='1h'))
    return df

# ----------------------------------------------------------------------
# 1. Regime Filter Tests
# ----------------------------------------------------------------------

class TestRegimeFilter:
    def setup_method(self):
        cfg = RegimeFilterConfig(
            adx_period=14,
            hurst_period=100,
            trend_vote_threshold=3
        )
        pos_cfg = PositionSizingConfig() # Default tiers
        self.rf = RegimeFilter(cfg, pos_cfg.tiers)

    def test_regime_range_market(self, mock_ohlcv_range):
        """测试震荡市识别"""
        df = mock_ohlcv_range
        res = self.rf.get_market_regime(df)
        assert res.regime == MarketRegime.RANGE
        assert res.position_ratio > 0.0

    def test_regime_trend_market(self, mock_ohlcv_trend):
        """测试趋势市识别"""
        df = mock_ohlcv_trend
        res = self.rf.get_market_regime(df)
        # 强趋势可能触发 TREND 或 FUSE
        assert res.regime in [MarketRegime.TREND, MarketRegime.FUSE]
        if res.regime == MarketRegime.TREND:
            assert res.position_ratio == 0.0

    def test_scan_dataframe_full(self, mock_ohlcv_range):
        """测试全量扫描功能"""
        results = self.rf.scan_dataframe_full(mock_ohlcv_range)
        assert len(results) == len(mock_ohlcv_range)
        assert results[0] is None
        assert isinstance(results[-1], RegimeResult)

# ----------------------------------------------------------------------
# 2. Grid Engine Tests
# ----------------------------------------------------------------------

class TestGridEngine:
    def setup_method(self):
        self.cfg = GridEngineConfig(
            ma_long_period=20, # Shorten for test
            ma_mid_period=10,
            grid_levels=5,
            min_order_qty=0.0001
        )
        self.ge = GridEngine(self.cfg)

    def test_grid_calculation_structure(self, mock_ohlcv_range):
        df = mock_ohlcv_range
        balance = 10000.0
        plan = self.ge.calculate_grid(df, total_balance=balance, position_ratio=1.0)
        
        assert "mid_price" in plan
        assert "buy_orders" in plan
        assert "sell_orders" in plan
        assert "capital_utilization" in plan
        assert len(plan["buy_orders"]) <= self.cfg.grid_levels

    def test_grid_capital_utilization(self, mock_ohlcv_range):
        """测试资金利用率随 Hurst 调节"""
        df = mock_ohlcv_range
        balance = 10000.0
        
        plan_full = self.ge.calculate_grid(df, total_balance=balance, position_ratio=1.0)
        cap_full = plan_full["capital_utilization"]
        
        plan_half = self.ge.calculate_grid(df, total_balance=balance, position_ratio=0.5)
        cap_half = plan_half["capital_utilization"]
        
        assert abs(cap_half - cap_full * 0.5) < 0.01


class TestBacktestEngine:
    def setup_method(self):
        self.bt = BacktestEngine()

    def _make_tick(self, ts, price, bar_ts=None):
        return TickData(timestamp=ts, price=price, bar_timestamp=bar_ts or ts)

    def test_match_with_ticks_buy(self):
        """价格下行时，路径内的买单应被成交"""
        self.bt.orders = [{'id': 1, 'side': 'buy', 'price': 100.0, 'qty': 1.0}]
        self.bt.balance_usdt = 200.0
        self.bt.balance_btc  = 0.0

        # 价格从 105 下行至 98，路径经过 100 → 买单应成交
        ticks = [
            self._make_tick(0, 105.0),
            self._make_tick(1, 98.0),
        ]
        self.bt._match_with_ticks(ticks)

        assert not self.bt.orders          # 挂单已清除
        assert len(self.bt.trades) == 1    # 生成一条成交记录
        assert self.bt.balance_usdt < 200.0
        assert self.bt.balance_btc > 0.0

    def test_match_with_ticks_sell(self):
        """价格上行时，路径内的卖单应被成交"""
        self.bt.orders = [{'id': 2, 'side': 'sell', 'price': 200.0, 'qty': 1.0}]
        self.bt.balance_btc  = 1.0
        self.bt.balance_usdt = 0.0

        # 价格从 190 上行至 210，路径经过 200 → 卖单应成交
        ticks = [
            self._make_tick(0, 190.0),
            self._make_tick(1, 210.0),
        ]
        self.bt._match_with_ticks(ticks)

        assert not self.bt.orders
        assert len(self.bt.trades) == 1
        assert self.bt.balance_btc < 1.0
        assert self.bt.balance_usdt > 0.0

    def test_match_with_ticks_no_false_fill(self):
        """价格从 105→98 时，买单 96 不在路径内，不应被成交"""
        self.bt.orders = [{'id': 3, 'side': 'buy', 'price': 96.0, 'qty': 1.0}]
        self.bt.balance_usdt = 200.0
        self.bt.balance_btc  = 0.0

        ticks = [
            self._make_tick(0, 105.0),
            self._make_tick(1, 98.0),   # 只下行到 98，未到 96
        ]
        self.bt._match_with_ticks(ticks)

        assert len(self.bt.orders) == 1  # 挂单仍在
        assert len(self.bt.trades) == 0  # 无成交

    def test_backtest_run_integration(self, mock_ohlcv_range):
        """集成测试：完整回测流程，验证接口和报告结构"""
        df   = mock_ohlcv_range
        feed = BacktestFeed(df, timeframe="1h")
        report = self.bt.run(feed, df)

        assert "metrics"      in report
        assert "trades"       in report
        assert "equity_curve" in report

        metrics = report["metrics"]
        assert metrics["trade_count"] >= 0
        assert isinstance(metrics["total_return_pct"], float)
        assert isinstance(metrics["final_usdt"], float)
        assert isinstance(metrics["final_btc"],  float)
        # 权益曲线每根收线 K 线都有一个快照（跳过预热期）
        assert len(report["equity_curve"]) > 0

