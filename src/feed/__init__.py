"""
src/feed - 数据馈送层

统一封装 Backtest（历史数据重放）和 Live（WebSocket 实时推送）的数据接口。
未来切换数据源只需替换 Feed 实现，Strategy / Engine 层无需改动。
"""
from .base import BarData, TickData, DataFeed
from .backtest_feed import BacktestFeed

__all__ = ["BarData", "TickData", "DataFeed", "BacktestFeed"]
