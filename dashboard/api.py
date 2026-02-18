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
from typing import Optional, List

# 确保项目根目录在 Python 路径中
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from src.config import Config
from src.exchange import ExchangeClient
from src.regime_filter import RegimeFilter, RegimeFilterConfig, MarketRegime
from src.strategy_config import load_strategy_config
from src.logger import setup_logger

logger = setup_logger("dashboard")

app = FastAPI(title="Grid Strategy Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# ======================================================================
# Pydantic 请求/响应模型
# ======================================================================

class RegimeParams(BaseModel):
    """前端传入的 Regime Filter 参数（全部可选，未传则使用 YAML 默认值）。"""
    adx_period: int = Field(default=14, ge=2)
    adx_trend_threshold: float = Field(default=25.0, gt=0, le=100)
    atr_short_period: int = Field(default=7, ge=2)
    atr_long_period: int = Field(default=28, ge=2)
    atr_ratio_threshold: float = Field(default=1.5, gt=0)
    bb_period: int = Field(default=20, ge=2)
    bb_std: float = Field(default=2.0, gt=0)
    bb_width_threshold: float = Field(default=0.1, gt=0)
    hurst_period: int = Field(default=100, ge=20)
    hurst_threshold: float = Field(default=0.5, gt=0, lt=1)
    trend_vote_threshold: int = Field(default=2, ge=1, le=4)


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
    trend_votes: int
    total_votes: int
    regime: str


class RegimeResponse(BaseModel):
    candles: List[CandleData]
    segments: List[RegimeSegment]
    latest: IndicatorSnapshot
    mode: str
    symbol: str
    timeframe: str


# ======================================================================
# 辅助函数
# ======================================================================

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
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 2. 拉取 K 线
    try:
        exchange = _get_exchange(req.mode)
        df = exchange.fetch_ohlcv(
            symbol=req.symbol,
            timeframe=req.timeframe,
            since_days=req.since_days,
        )
    except Exception as e:
        logger.error("K 线数据拉取失败: %s", e)
        raise HTTPException(status_code=503, detail=f"数据拉取失败: {e}")

    if len(df) < rf_config.hurst_period:
        raise HTTPException(
            status_code=400,
            detail=f"数据不足：需要至少 {rf_config.hurst_period} 根 K 线，当前只有 {len(df)} 根。"
        )

    # 3. 计算 Regime（滚动窗口）
    rf = RegimeFilter(config=rf_config)
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
        trend_votes=latest_result.trend_votes,
        total_votes=latest_result.total_votes,
        regime=latest_result.regime.value,
    )

    return RegimeResponse(
        candles=candles,
        segments=segments,
        latest=latest,
        mode=req.mode,
        symbol=req.symbol,
        timeframe=req.timeframe,
    )


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """返回前端 HTML 页面。"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="前端文件未找到")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
