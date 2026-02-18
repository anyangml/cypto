"""
exchange.py - 交易所连接层

封装 CCXT 对 Binance 的连接，支持 Testnet 和实盘模式切换。
提供账户余额查询和 K 线数据获取功能。
"""

from __future__ import annotations

import time
from typing import Optional

import ccxt
import pandas as pd

from src.config import Config
from src.logger import setup_logger

logger = setup_logger("exchange")

# Binance Testnet 的 REST API 地址
_TESTNET_URLS = {
    "api": "https://testnet.binance.vision/api",
}


class ExchangeClient:
    """
    Binance 交易所客户端。

    封装 CCXT 的 Binance 实例，统一处理 Testnet/Live 模式切换、
    错误重试和日志记录。
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._exchange = self._build_exchange()
        # 初始化一个仅用于获取行情的合约交易所实例（无需 API Key）
        self._futures_exchange = self._build_futures_exchange()
        logger.info(
            "ExchangeClient 初始化完成 | 模式: %s | 交易对: %s",
            config.trading_mode.upper(),
            config.symbol,
        )

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _build_exchange(self) -> ccxt.binance:
        """根据配置构建 CCXT Binance 实例。"""
        params: dict = {
            "apiKey": self._config.binance_api_key,
            "secret": self._config.binance_api_secret,
            "enableRateLimit": True,  # 自动遵守 Binance 频率限制
            "options": {
                "defaultType": "spot",  # 默认现货交易
            },
        }

        if self._config.is_testnet:
            # 切换到 Binance Testnet 端点
            params["urls"] = {"api": _TESTNET_URLS}
            logger.warning("当前运行在 TESTNET 模式，不会产生真实交易。")

        return ccxt.binance(params)

    def _build_futures_exchange(self) -> ccxt.binance:
        """构建用于获取合约数据的只读实例。"""
        params = {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
        if self._config.is_testnet:
             params["urls"] = {"api": _TESTNET_URLS}
        return ccxt.binance(params)

    def _safe_call(self, func, *args, retries: int = 3, delay: float = 2.0, **kwargs):
        """
        带重试机制的 API 调用包装器。

        Args:
            func:    要调用的 CCXT 方法。
            retries: 最大重试次数。
            delay:   每次重试前的等待秒数（指数退避）。

        Returns:
            API 调用结果。

        Raises:
            ccxt.BaseError: 超过最大重试次数后仍失败时抛出。
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                return func(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                wait = delay * attempt
                logger.warning("触发频率限制，%.1f 秒后重试 (%d/%d)...", wait, attempt, retries)
                time.sleep(wait)
                last_exc = e
            except ccxt.NetworkError as e:
                wait = delay * attempt
                logger.warning("网络错误: %s，%.1f 秒后重试 (%d/%d)...", e, wait, attempt, retries)
                time.sleep(wait)
                last_exc = e
            except ccxt.AuthenticationError as e:
                logger.error("API 认证失败，请检查 API Key/Secret: %s", e)
                raise
            except ccxt.BaseError as e:
                logger.error("交易所 API 错误: %s", e)
                raise

        logger.error("API 调用失败，已达最大重试次数 (%d)。", retries)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def fetch_balance(self) -> dict:
        """
        获取账户余额。
        
        注意：目前返回的是 'Total' 余额（即 可用余额 + 挂单锁定金额）。
        如果与网页端显示不一致，请检查是否有未成交的挂单。

        Returns:
            资产总量字典，例如: {"BTC": 0.5, "USDT": 1000.0}
        """
        raw = self._safe_call(self._exchange.fetch_balance)
        
        # 从原始数据中提取 'total' (total = free + used)
        # 仅过滤掉余额为 0 的资产
        balances = {
            asset: info
            for asset, info in raw["total"].items()
            if isinstance(info, (int, float)) and info > 0
        }
        
        logger.info("账户[总余额]查询成功 (包含可用与锁定资产): %s", balances)
        return balances

    def fetch_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        since_days: int = 30,
        since_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        获取历史 K 线数据 (OHLCV)。

        Args:
            symbol:     交易对，默认使用 Config 中的配置。
            timeframe:  K 线周期，默认使用 Config 中的配置。
            since_days: 获取过去多少天的数据（若未指定 since_ms）。
            since_ms:   确定的起始时间戳（毫秒），若指定则忽略 since_days。
            limit:      单次请求最大条数（Binance 上限 1000）。

        Returns:
            包含 [timestamp, open, high, low, close, volume] 列 a DataFrame，
            timestamp 列已转换为 UTC datetime 类型并设为索引。
        """
        symbol = symbol or self._config.symbol
        timeframe = timeframe or self._config.timeframe

        # 计算起始时间戳（或者使用传入的 since_ms）
        if since_ms is None:
            since_ms = int(
                (pd.Timestamp.now("UTC") - pd.Timedelta(days=since_days)).timestamp() * 1000
            )

        logger.info(
            "开始获取 K 线数据 | 交易对: %s | 周期: %s | 起始时间戳: %s",
            symbol,
            timeframe,
            since_ms,
        )

        all_candles: list = []
        current_since = since_ms

        while True:
            candles = self._safe_call(
                self._exchange.fetch_ohlcv,
                symbol,
                timeframe,
                since=current_since,
                limit=limit,
            )
            if not candles:
                break

            all_candles.extend(candles)

            # 若返回数量小于 limit，说明已到达最新数据
            if len(candles) < limit:
                break

            # 移动时间窗口，避免重复获取
            current_since = candles[-1][0] + 1

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)

        # 去重（分页请求可能产生重复）
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        logger.info(
            "K 线数据获取完成 | 共 %d 条 | 时间范围: %s ~ %s",
            len(df),
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
        )
        return df

    def save_ohlcv_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        将 K 线 DataFrame 保存为 CSV 文件。

        Args:
            df:       fetch_ohlcv 返回的 DataFrame。
            filepath: 保存路径（含文件名）。
        """
        df.to_csv(filepath)
        logger.info("K 线数据已保存至: %s", filepath)

    def create_limit_order(self, side: str, price: float, quantity: float) -> dict:
        """
        发送限价单。

        Args:
            side:     "buy" 或 "sell"。
            price:    委托价格。
            quantity: 委托数量。

        Returns:
            CCXT 订单对象 (dict)，包含 id, status 等信息。
        """
        # 价格/数量精度修正（简单处理，建议在 execution 层做更严格的 stepSize 检查）
        price = float(price)
        quantity = float(quantity)

        logger.info("正在提交限价单: %s %s @ %s", side.upper(), quantity, price)
        try:
            order = self._safe_call(
                self._exchange.create_order,
                symbol=self._config.symbol,
                type="limit",
                side=side,
                amount=quantity,
                price=price,
            )
            logger.info("下单成功: ID=%s, Status=%s", order.get("id"), order.get("status"))
            return order
        except Exception as e:
            logger.error("下单失败: %s", e)
            raise

    def cancel_order(self, order_id: str) -> dict:
        """
        撤销指定订单。
        """
        logger.info("正在撤销订单: %s", order_id)
        try:
            return self._safe_call(
                self._exchange.cancel_order,
                id=order_id,
                symbol=self._config.symbol,
            )
        except Exception as e:
            logger.error("撤单失败:ID=%s, Error=%s", order_id, e)
            raise

    def fetch_open_orders(self) -> list:
        """
        获取当前交易对的所有挂单。
        """
        orders = self._safe_call(
            self._exchange.fetch_open_orders,
            symbol=self._config.symbol,
        )
        logger.debug("查询到 %d 个挂单", len(orders))
        return orders

    def cancel_all_orders(self) -> list:
        """
        撤销当前交易对的所有挂单。
        """
        logger.info("正在撤销所有挂单...")
        try:
            # 尝试使用 CCXT 的 cancel_all_orders (如果交易所支持)
            # Binance 支持此接口
            return self._safe_call(
                self._exchange.cancel_all_orders,
                symbol=self._config.symbol,
            )
        except Exception as e:
            # 如果不支持或失败，回退到逐个撤单
            logger.warning("批量撤单接口调用失败，尝试逐个撤单: %s", e)
            open_orders = self.fetch_open_orders()
            results = []
            for order in open_orders:
                try:
                    res = self.cancel_order(order["id"])
                    results.append(res)
                except Exception:
                    pass
            return results

    def fetch_open_interest(self) -> float:
        """
        获取当前合约的持仓量 (Open Interest)。
        注意：即使是现货策略，也参考合约的 OI 作为风控指标。
        """
        try:
            # OI 数据通常以合约为准
            # 现货 Symbol (BTC/USDT) 在合约中通常也是 BTC/USDT
            res = self._futures_exchange.fetch_open_interest(self._config.symbol)
            # res 结构: {'symbol': 'BTC/USDT', 'openInterestAmount': 123.4, 'openInterestValue': ...}
            # 我们关注数量还是价值？通常关注 Value (USDT)
            # 但 CCXT openInterestAmount 是币的数量。
            # 为了风控，关注 Amount 的变化率即可。
            return float(res.get("openInterestAmount", 0.0))
        except Exception as e:
            logger.error("获取 Open Interest 失败: %s", e)
            return 0.0

    def fetch_funding_rate(self) -> float:
        """
        获取当前合约的资金费率。
        """
        try:
            res = self._futures_exchange.fetch_funding_rate(self._config.symbol)
            # res: {'fundingRate': 0.0001, ...}
            return float(res.get("fundingRate", 0.0))
        except Exception as e:
            logger.error("获取资金费率失败: %s", e)
            return 0.0
