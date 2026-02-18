"""
tests/test_ticket001.py - Ticket-001 验收测试 (Pydantic 版)

验收标准:
  AC1: 成功连接币安 API 并能打印当前账户余额。
  AC2: 能够下载过去 30 天的 BTC/USDT 1h K 线数据并保存为 CSV。
  AC3: 统一的日志系统能够记录 INFO 和 ERROR 级别信息。
"""

import os
import logging
import pytest
import pandas as pd
from pydantic import ValidationError

from src.config import load_config, Config, TradingMode
from src.logger import setup_logger
from src.exchange import ExchangeClient


# ======================================================================
# AC3: 日志系统测试
# ======================================================================

class TestLogger:
    def test_logger_creates_instance(self):
        log = setup_logger("test_logger")
        assert isinstance(log, logging.Logger)

    def test_logger_has_handlers(self):
        log = setup_logger("test_logger_handlers")
        assert len(log.handlers) >= 1

    def test_logger_info_level(self, caplog):
        log = setup_logger("test_info")
        with caplog.at_level(logging.INFO, logger="test_info"):
            log.info("这是一条 INFO 测试日志")
        assert "这是一条 INFO 测试日志" in caplog.text


# ======================================================================
# 配置模块测试 (Pydantic)
# ======================================================================

class TestConfig:
    def test_config_invalid_trading_mode(self, monkeypatch):
        """无效的 TRADING_MODE 应抛出 ValidationError。"""
        monkeypatch.setenv("TRADING_MODE", "invalid_mode")
        monkeypatch.setenv("BINANCE_API_KEY", "real_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "real_secret")
        with pytest.raises(ValidationError):
            load_config()

    def test_config_missing_api_key(self, monkeypatch):
        """占位符 API Key 应抛出 ValidationError。"""
        monkeypatch.setenv("BINANCE_API_KEY", "your_api_key_here")
        monkeypatch.setenv("BINANCE_API_SECRET", "real_secret")
        monkeypatch.setenv("TRADING_MODE", "testnet")
        with pytest.raises(ValidationError):
            load_config()

    def test_config_is_testnet_flag(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "real_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "real_secret")
        
        monkeypatch.setenv("TRADING_MODE", "testnet")
        config = load_config()
        assert config.is_testnet is True

        monkeypatch.setenv("TRADING_MODE", "live")
        config2 = load_config()
        assert config2.is_testnet is False


# ======================================================================
# AC1 & AC2: 交易所连接测试 (需要真实 API Key)
# ======================================================================

@pytest.mark.integration
class TestExchangeClient:
    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            self.config = load_config()
            self.client = ExchangeClient(self.config)
        except (ValidationError, ValueError) as e:
            pytest.skip(f"跳过集成测试（API Key 未配置或无效）: {e}")

    def test_ac1_fetch_balance(self):
        balance = self.client.fetch_balance()
        assert isinstance(balance, dict)
        assert len(balance) > 0

    def test_ac2_fetch_ohlcv_returns_dataframe(self):
        df = self.client.fetch_ohlcv(since_days=30)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) >= 600

    def test_ac2_save_ohlcv_to_csv(self, tmp_path):
        df = self.client.fetch_ohlcv(since_days=1)
        csv_path = str(tmp_path / "btc_usdt_1h.csv")
        self.client.save_ohlcv_to_csv(df, csv_path)
        assert os.path.isfile(csv_path)
