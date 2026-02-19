"""
src/feed/live_feed.py - 实时数据馈送（占位存根，待实现）

未来集成 Binance WebSocket 时在此实现，无需修改 BacktestEngine 或 Strategy 层。

计划实现方案：
  · 数据源 1（K 线）：wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}
    - is_closed=False 的 BarData → 只更新 Dashboard 图表，不触发策略
    - is_closed=True  的 BarData → 触发 Engine.on_bar()，同时生成 TickData

  · 数据源 2（精细撮合）：wss://stream.binance.com:9443/ws/{symbol}@aggTrade
    - 每笔真实成交 → 直接作为 TickData 送入订单簿，替代 BacktestFeed 的合成路径
    - 撮合精度达到逐笔级别

接口契约（必须实现才能替换 BacktestFeed）：
    def iter_bars(self) -> Iterator[Tuple[BarData, List[TickData]]]:
        ...

实现时参考的 ccxt.pro（异步 WebSocket 封装）：
    import ccxt.pro as ccxtpro
    exchange = ccxtpro.binance()
    async for ohlcv in exchange.watch_ohlcv("BTC/USDT", "1m"):
        ...

注意事项：
  · LiveFeed.iter_bars() 应是异步生成器（AsyncIterator），需调整 Engine 为 asyncio
  · 或者在独立线程运行 WebSocket，通过 asyncio.Queue 传递给同步 Engine
  · 记录 is_closed=False 的 bar 以便 Dashboard 实时更新（SSE 推送）
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

from .base import BarData, TickData


class LiveFeed:
    """
    实时数据馈送（未实现，接口存根）。

    实现 DataFeed 协议，未来替换 BacktestFeed 时引擎层零改动。
    """

    def __init__(self, symbol: str, timeframe: str = "1m") -> None:
        self._symbol = symbol
        self._timeframe = timeframe

    def iter_bars(self) -> Iterator[Tuple[BarData, List[TickData]]]:
        raise NotImplementedError(
            "LiveFeed 尚未实现。\n"
            "请参考本文件顶部注释中的 WebSocket 集成方案。"
        )
