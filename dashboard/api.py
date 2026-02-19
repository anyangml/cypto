"""
dashboard/api.py - 策略可视化面板后端 API

提供以下 REST 端点：
  GET  /api/klines          - 获取 K 线数据（回测/实时模式）
  POST /api/regime          - 计算 Regime Filter 指标（支持自定义参数）
  GET  /api/config          - 获取当前策略配置
  GET  /                    - 返回前端 HTML 页面
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
from functools import lru_cache

# 确保项目根目录在 Python 路径中
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from src.config import Config
from src.exchange import ExchangeClient, timeframe_to_ms
from src.regime_filter import RegimeFilter, RegimeFilterConfig, MarketRegime
from src.strategy_config import load_strategy_config
from src.logger import setup_logger
from src.backtest import BacktestEngine
from src.grid_engine import GridEngine
from src.database import DatabaseHandler
from src.feed import BacktestFeed

logger = setup_logger("dashboard")

# 网格计划预览时使用的估算权益（USDT）。
# Live 模式未来应从 ExchangeClient.fetch_balance() 读取真实账户权益。
# 与 BacktestEngine._initial_capital 默认值保持一致。
_DEFAULT_GRID_EQUITY: float = 10_000.0

app = FastAPI(title="Grid Strategy Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# 挂载静态文件目录
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# ======================================================================
# Data Manager (Caching & Incremental Fetching)
# ======================================================================

class DataManager:
    """管理 K 线数据的缓存与增量更新。"""
    def __init__(self):
        # key: (mode, symbol, timeframe) -> value: pd.DataFrame
        self._cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    # K 线根数上限；超过该数量会使 Regime 扫描非常缓慢
    _MAX_CANDLES = 10_000

    def get_ohlcv(
        self,
        mode: str,
        symbol: str,
        timeframe: str,
        since_days: int
    ) -> pd.DataFrame:
        # ---- 自动收窄 since_days，避免低 timeframe 时拉取海量数据 ----
        tf_ms = timeframe_to_ms(timeframe)
        max_days = max(1, int(self._MAX_CANDLES * tf_ms / 86_400_000))
        if since_days > max_days:
            logger.info(
                "since_days=%d 超过 %s 的合理上限 %d 天（约 %d 根 K 线），已自动收窄",
                since_days, timeframe, max_days, self._MAX_CANDLES,
            )
            since_days = max_days
        # -------------------------------------------------------------

        key = (mode, symbol, timeframe)
        now = pd.Timestamp.utcnow()
        required_start_dt = now - pd.Timedelta(days=since_days)
        required_start_ms = int(required_start_dt.timestamp() * 1000)

        # 检查缓存是否可用
        if key in self._cache:
            df_cached = self._cache[key]
            cached_start_ms = int(df_cached.index[0].timestamp() * 1000)
            cached_end_ms   = int(df_cached.index[-1].timestamp() * 1000)

            # 缓存覆盖所需起始时间 → 增量更新
            if cached_start_ms <= required_start_ms:
                logger.debug("使用缓存进行增量更新: %s", key)
                try:
                    exchange = _get_exchange(mode)
                    df_new = exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since_ms=cached_end_ms,
                    )
                    if not df_new.empty:
                        df_combined = pd.concat([df_cached, df_new])
                        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
                        df_combined.sort_index(inplace=True)
                        self._cache[key] = df_combined
                        return df_combined[df_combined.index >= required_start_dt]
                    else:
                        return df_cached[df_cached.index >= required_start_dt]
                except Exception as e:
                    logger.warning("增量更新失败，回退到全量拉取: %s", e)

        # 全量拉取
        logger.info("全量拉取数据: %s, days=%d", key, since_days)
        exchange = _get_exchange(mode)
        df = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since_days=since_days,
        )
        self._cache[key] = df
        return df

# 全局单例
data_manager = DataManager()


# ======================================================================
# Pydantic 请求/响应模型
# ======================================================================

class RegimeParams(BaseModel):
    """前端传入的 Regime Filter 参数（全部可选，未传则使用 YAML 默认值）。"""
    adx_period: int = Field(default=14, ge=2)
    adx_trend_threshold: float = Field(default=25.0, gt=0, le=100)
    adx_fuse_threshold: float = Field(default=30.0, gt=0, le=100)
    atr_short_period: int = Field(default=7, ge=2)
    atr_long_period: int = Field(default=28, ge=2)
    atr_ratio_threshold: float = Field(default=1.5, gt=0)
    bb_period: int = Field(default=20, ge=2)
    bb_std: float = Field(default=2.0, gt=0)
    bb_width_threshold: float = Field(default=0.1, gt=0)
    hurst_period: int = Field(default=100, ge=20)
    hurst_threshold: float = Field(default=0.5, gt=0, lt=1)
    hurst_fuse_threshold: float = Field(default=0.6, gt=0, lt=1)
    hurst_extreme_low: float = Field(default=0.35, gt=0, lt=1)
    ma200_period: int = Field(default=200, ge=20)
    ma200_slope_lookback: int = Field(default=10, ge=1)
    ma200_slope_threshold: float = Field(default=0.0)
    trend_vote_threshold: int = Field(default=2, ge=1, le=5)


class RegimeRequest(BaseModel):
    """POST /api/regime 的请求体。"""
    mode: str = Field(default="backtest", pattern="^(backtest|live)$")
    symbol: str = Field(default="BTC/USDT")
    timeframe: str = Field(default="1h")
    since_days: int = Field(default=90, ge=1, le=365)
    params: RegimeParams = RegimeParams()


class CandleData(BaseModel):
    time: int       # Unix timestamp (seconds)
    open: float
    high: float
    low: float
    close: float
    volume: float


class RegimeSegment(BaseModel):
    """K 线图上的 Regime 着色区间。"""
    start_time: int
    end_time: int
    regime: str     # "RANGE" or "TREND"


class IndicatorSnapshot(BaseModel):
    """最新时间点的各指标数值快照。"""
    adx_value: float
    adx_vote: bool
    atr_ratio: float
    atr_vote: bool
    bb_width: float
    bb_vote: bool
    hurst_value: float
    hurst_vote: bool
    ma200_slope: float
    ma200_vote: bool
    trend_votes: int
    total_votes: int
    regime: str
    position_ratio: float
    accumulate_spot: bool
    fuse_triggered: bool


class BacktestOutput(BaseModel):
    equity_curve: List[Dict]
    trades: List[Dict]
    metrics: Dict

class GridPlan(BaseModel):
    mid_price: float
    upper_boundary: float
    lower_boundary: float
    capital_utilization: float
    active_capital: float
    buy_orders: List[Dict]
    sell_orders: List[Dict]
    overbought_triggered: bool
    downtrend_protection: bool

class RegimeResponse(BaseModel):
    candles: List[CandleData]
    segments: List[RegimeSegment]
    latest: IndicatorSnapshot
    mode: str
    symbol: str
    timeframe: str
    backtest: Optional[BacktestOutput] = None
    grid_plan: Optional[GridPlan] = None


# ======================================================================
# 辅助函数
# ======================================================================

@lru_cache(maxsize=2)
def _get_exchange(mode: str) -> ExchangeClient:
    """根据模式创建对应的交易所客户端。"""
    try:
        cfg = Config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"配置加载失败: {e}")

    # 回测模式强制使用 testnet，实时模式使用配置文件中的设置
    if mode == "backtest":
        from src.config import TradingMode
        import os
        # 临时覆盖为 testnet
        original = os.environ.get("TRADING_MODE")
        os.environ["TRADING_MODE"] = "testnet"
        try:
            cfg = Config()
        finally:
            if original is None:
                os.environ.pop("TRADING_MODE", None)
            else:
                os.environ["TRADING_MODE"] = original

    return ExchangeClient(cfg)


def _build_regime_segments(
    timestamps: list, regimes: list
) -> List[RegimeSegment]:
    """将逐 K 线的 Regime 序列合并为连续区间，减少前端渲染压力。"""
    segments = []
    if not regimes:
        return segments

    start_idx = 0
    current_regime = regimes[0]

    for i in range(1, len(regimes)):
        if regimes[i] != current_regime or i == len(regimes) - 1:
            end_idx = i if regimes[i] != current_regime else i
            segments.append(RegimeSegment(
                start_time=int(timestamps[start_idx]),
                end_time=int(timestamps[end_idx - 1]),
                regime=current_regime,
            ))
            start_idx = i
            current_regime = regimes[i]

    # 最后一段
    if start_idx < len(regimes):
        segments.append(RegimeSegment(
            start_time=int(timestamps[start_idx]),
            end_time=int(timestamps[-1]),
            regime=current_regime,
        ))

    return segments


# ======================================================================
# API 路由
# ======================================================================

@app.get("/api/config")
def get_config():
    """返回当前 strategy.yaml 中的配置。"""
    try:
        cfg = load_strategy_config()
        return cfg.regime_filter.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/regime", response_model=RegimeResponse)
def compute_regime(req: RegimeRequest):
    """
    核心计算端点：
    1. 根据 mode 拉取 K 线数据
    2. 用传入的参数实例化 RegimeFilter
    3. 对每根 K 线计算 Regime 状态
    4. 返回 K 线数据 + Regime 区间 + 最新指标快照
    """
    # 1. 构建 RegimeFilterConfig（Pydantic 自动校验）
    try:
        rf_config = RegimeFilterConfig(**req.params.model_dump())
    except Exception as e:
        logger.error("参数校验失败: %s", e)
        # 如果是 Pydantic 校验错误，返回详细信息
        if hasattr(e, 'errors'):
             raise HTTPException(status_code=422, detail=e.errors())
        raise HTTPException(status_code=422, detail=str(e))

    # 2. 拉取 K 线（使用 DataManager 进行增量/缓存处理）
    try:
        df = data_manager.get_ohlcv(
            mode=req.mode,
            symbol=req.symbol,
            timeframe=req.timeframe,
            since_days=req.since_days,
        )
    except Exception as e:
        logger.error("K 线数据获取失败: %s", e)
        raise HTTPException(status_code=503, detail=f"数据获取失败: {e}")

    if len(df) < rf_config.hurst_period:
        raise HTTPException(
            status_code=400,
            detail=f"数据不足：需要至少 {rf_config.hurst_period} 根 K 线，当前只有 {len(df)} 根。"
        )

    # 3. 计算 Regime（滚动窗口）
    # 尝试加载完整策略配置以获取 tiers
    tiers = []
    try:
        strategy_cfg = load_strategy_config()
        tiers = strategy_cfg.position_sizing.tiers
    except Exception as e:
        logger.warning("策略配置加载失败，将使用默认分级逻辑: %s", e)

    rf = RegimeFilter(config=rf_config, tiers=tiers)
    min_required = max(
        rf_config.adx_period * 2,
        rf_config.atr_long_period * 2,
        rf_config.bb_period,
        rf_config.hurst_period,
    )

    timestamps_sec = []
    regime_labels = []
    latest_result = None

    for i in range(min_required, len(df) + 1):
        window = df.iloc[:i]
        try:
            result = rf.get_market_regime(window)
            ts = int(window.index[-1].timestamp())
            timestamps_sec.append(ts)
            regime_labels.append(result.regime.value)
            latest_result = result
        except Exception:
            pass

    if latest_result is None:
        raise HTTPException(status_code=500, detail="Regime 计算失败，请检查参数配置。")

    # 4. 构建 K 线响应
    candles = [
        CandleData(
            time=int(ts.timestamp()),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )
        for ts, row in df.iterrows()
    ]

    # 5. 构建 Regime 区间
    segments = _build_regime_segments(timestamps_sec, regime_labels)

    # 6. 最新指标快照
    latest = IndicatorSnapshot(
        adx_value=round(latest_result.adx_value, 2),
        adx_vote=latest_result.adx_vote,
        atr_ratio=round(latest_result.atr_ratio, 4),
        atr_vote=latest_result.atr_vote,
        bb_width=round(latest_result.bb_width, 4),
        bb_vote=latest_result.bb_vote,
        hurst_value=round(latest_result.hurst_value, 4),
        hurst_vote=latest_result.hurst_vote,
        ma200_slope=round(latest_result.ma200_slope, 5),
        ma200_vote=latest_result.ma200_vote,
        trend_votes=latest_result.trend_votes,
        total_votes=latest_result.total_votes,
        regime=latest_result.regime.value,
        position_ratio=round(latest_result.position_ratio, 2),
        accumulate_spot=latest_result.accumulate_spot,
        fuse_triggered=latest_result.fuse_triggered,
    )

    # 7. 计算最新时刻的网格计划 (用于前端可视化)
    grid_plan = None
    try:
        ge         = GridEngine(load_strategy_config().grid_engine)
        # Live 模式应从交易所获取真实权益；回测/预览时用配置默认值
        est_equity = _DEFAULT_GRID_EQUITY
        gp         = ge.calculate_grid(df, est_equity, latest_result.position_ratio)
        grid_plan  = GridPlan(**gp)
    except Exception as e:
        logger.warning("网格计划计算失败: %s", e)

    # 8. 如果是回测模式，执行回测模拟
    backtest_result = None
    if req.mode == "backtest":
        full_config = load_strategy_config()
        full_config.regime_filter = rf_config  # 使用前端实时调节的参数

        bt   = BacktestEngine(strategy_cfg=full_config)
        feed = BacktestFeed(df, timeframe=req.timeframe)
        try:
            raw_res = bt.run(feed, df)
            backtest_result = BacktestOutput(**raw_res)
        except Exception as e:
            logger.error("回测执行失败: %s", e)

    return RegimeResponse(
        candles=candles,
        segments=segments,
        latest=latest,
        mode=req.mode,
        symbol=req.symbol,
        timeframe=req.timeframe,
        backtest=backtest_result,
        grid_plan=grid_plan
    )


# 数据库实例
db_handler = DatabaseHandler()

@app.get("/api/history/trades")
def get_history_trades(limit: int = 100):
    """获取历史成交记录"""
    return db_handler.get_trades(limit=limit)

@app.get("/api/history/stats")
def get_history_stats():
    """获取历史统计数据"""
    return db_handler.get_stats()

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """返回前端 HTML 页面。"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="前端文件未找到")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

# Resolve deferred types for Pydantic compatible with future annotations
try:
    BacktestOutput.model_rebuild()
    GridPlan.model_rebuild()
    RegimeResponse.model_rebuild()
except Exception:
    pass
