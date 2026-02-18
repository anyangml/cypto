"""
config.py - 基于 Pydantic 的全局配置管理模块

通过 pydantic-settings 自动加载环境变量并进行严格校验。
"""

from enum import Enum
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    TESTNET = "testnet"
    LIVE = "live"


class Config(BaseSettings):
    """
    配置模型。
    自动从环境变量中读取（忽略大小写，例如 BINANCE_API_KEY -> api_key）。
    """

    # --- Binance API ---
    binance_api_key: str = Field(alias="binance_api_key")
    binance_api_secret: str = Field(alias="binance_api_secret")

    # --- Trading Mode ---
    trading_mode: TradingMode = Field(default=TradingMode.TESTNET)

    # --- Logging ---
    log_level: str = Field(default="INFO")

    # --- Trading Parameters ---
    symbol: str = Field(default="BTC/USDT")
    timeframe: str = Field(default="1h")

    # --- Notification ---
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # 配置 Pydantic 行为
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # 忽略额外的环境变量
    )

    @property
    def is_testnet(self) -> bool:
        return self.trading_mode == TradingMode.TESTNET

    @field_validator("binance_api_key", "binance_api_secret")
    @classmethod
    def check_not_placeholder(cls, v: str) -> str:
        """检查是否为占位符字符串。"""
        placeholders = ["your_api_key_here", "your_api_secret_here", ""]
        if v.lower() in placeholders:
            raise ValueError("API Key/Secret 未配置，请在 .env 文件中填写真实值。")
        return v

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """确保 symbol 格式正确。"""
        if "/" not in v:
            raise ValueError(f"交易对格式错误: '{v}'。应为 'BTC/USDT' 格式。")
        return v.upper()


def load_config() -> Config:
    """实例化配置，会自动触发校验。"""
    return Config()
