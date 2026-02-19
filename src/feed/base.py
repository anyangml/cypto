"""
src/feed/base.py - 数据馈送抽象层

定义 Backtest 和 Live 两种模式共用的数据单元（BarData, TickData）
以及统一的 DataFeed 协议接口（Python structural subtyping）。

设计目标：
  - Backtest 和 Live 均通过 BarData / TickData 传递数据，Strategy 层无感知
  - 新增 LiveFeed 只需实现 iter_bars() 即可，无需继承任何基类
  - 冻结 dataclass 防止意外变动，保证线程安全
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class BarData:
    """
    标准 K 线数据单元 —— Backtest 和 Live 共用。

    Backtest: 来自历史 OHLCV DataFrame（is_closed 始终为 True）
    Live:     来自 Binance WebSocket kline stream
              · is_closed=False → 当前 K 线仍在进行，只更新图表
              · is_closed=True  → K 线已收盘，触发策略逻辑
    """
    timestamp: int    # Unix seconds，K 线收盘时间
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float
    timeframe: str    # "1m", "5m", "1h" ...
    is_closed: bool = True


@dataclass(frozen=True)
class TickData:
    """
    价格刻度 —— 用于精细化订单撮合。

    Backtest: 由 BacktestFeed 根据 OHLC 路径合成（每根 bar 4 个 waypoint）
    Live:     未来可来自 Binance aggTrade / bookTicker WebSocket stream

    timestamp 在 Backtest 中为 bar_timestamp + waypoint_offset（秒），
    保证同一根 bar 内的多笔成交有不同的时间戳，History 列表更真实。
    """
    timestamp:     int    # Unix seconds
    price:         float
    bar_timestamp: int    # 所属 K 线的时间戳，用于权益快照对齐


@runtime_checkable
class DataFeed(Protocol):
    """
    数据馈送协议（structural subtyping，不需要显式继承）。

    任何实现了 iter_bars() 的类均满足此协议，可直接传入 BacktestEngine。
    LiveFeed 在未来只需实现相同签名的 iter_bars() 即可无缝替换。
    """

    def iter_bars(self) -> Iterator[Tuple[BarData, List[TickData]]]:
        """
        逐一产出 (bar, ticks) 对：
          - bar:   已收线的 BarData（is_closed=True）
          - ticks: 该 bar 内按时间升序的价格刻度列表，首个 tick = Open 价
        """
        ...
