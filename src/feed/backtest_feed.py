"""
src/feed/backtest_feed.py - 历史数据馈送（回测专用）

将 OHLCV DataFrame 转化为 (BarData, List[TickData]) 序列，
供 BacktestEngine 消费，精度优于纯 OHLC 区间撮合。

合成 Tick 路径规则（基于 K 线方向的标准假设）：
  阳线 (close >= open): Open → Low → High → Close
  阴线 (close <  open): Open → High → Low → Close

时间戳分配：bar_timestamp + 0, 1, 2, 3 (秒偏移)
即使是 1m K 线也有 4 个不同时间戳，让 History 记录无重叠。

未来 LiveFeed 设计参考：
  LiveFeed.iter_bars() 消费 Binance WebSocket kline_1m stream，
  在 is_closed=True 时 yield 真实 BarData；
  aggTrade stream 可直接产生 TickData，替换合成路径。
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import pandas as pd

from .base import BarData, TickData

# 每根 K 线合成的价格刻度数量（Open + extreme_1 + extreme_2 + Close）
_WAYPOINTS_PER_BAR: int = 4


class BacktestFeed:
    """
    从历史 OHLCV DataFrame 生成 (BarData, List[TickData]) 序列。

    实现 DataFeed 协议，可直接传入任何依赖 DataFeed 的引擎：
        feed = BacktestFeed(df, timeframe="1m")
        engine.run(feed)

    设计约束：
      · 只读，不修改传入 df
      · iter_bars() 是惰性生成器，内存占用 O(1)
      · 所有计算在 _make_ticks() 中集中，方便单元测试
    """

    def __init__(self, df: pd.DataFrame, timeframe: str = "1h") -> None:
        """
        Args:
            df:        标准 OHLCV DataFrame，index 为 UTC datetime（来自 ExchangeClient.fetch_ohlcv）
            timeframe: K 线周期字符串，例如 "1m", "15m", "1h"
        """
        if df.empty:
            raise ValueError("BacktestFeed 收到空 DataFrame")
        self._df = df
        self._timeframe = timeframe

    # ------------------------------------------------------------------
    # DataFeed Protocol 实现
    # ------------------------------------------------------------------

    def iter_bars(self) -> Iterator[Tuple[BarData, List[TickData]]]:
        """逐行产出 (BarData, List[TickData])，已按时间升序排列。"""
        for ts_index, row in self._df.iterrows():
            bar = BarData(
                timestamp=int(ts_index.timestamp()),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                timeframe=self._timeframe,
                is_closed=True,
            )
            yield bar, self._make_ticks(bar)

    # ------------------------------------------------------------------
    # 合成 Tick 生成（可单独测试）
    # ------------------------------------------------------------------

    @staticmethod
    def _make_ticks(bar: BarData) -> List[TickData]:
        """
        根据 K 线 OHLC 方向生成 4 个合成价格刻度，模拟价格路径。

        阳线: Open → Low  → High → Close  （先探底，再冲高）
        阴线: Open → High → Low  → Close  （先冲高，再探底）

        时间戳 = bar.timestamp + waypoint_index（0-3 秒偏移）。
        同一 bar 内的多笔成交因此具有不同时间戳，消除"同时扫单"现象。
        """
        is_green = bar.close >= bar.open
        if is_green:
            prices = [bar.open, bar.low, bar.high, bar.close]
        else:
            prices = [bar.open, bar.high, bar.low, bar.close]

        return [
            TickData(
                timestamp=bar.timestamp + offset,
                price=price,
                bar_timestamp=bar.timestamp,
            )
            for offset, price in enumerate(prices)
        ]
