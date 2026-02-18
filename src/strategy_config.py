"""
src/strategy_config.py - 策略参数加载模块

从 config/strategy.yaml 中读取所有策略参数，
通过 Pydantic 进行自动校验，并返回强类型的配置对象。

使用方式:
    from src.strategy_config import load_strategy_config
    cfg = load_strategy_config()
    rf = RegimeFilter(config=cfg.regime_filter)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ValidationError

from src.regime_filter import RegimeFilterConfig
from src.logger import setup_logger

logger = setup_logger("strategy_config")

# 默认配置文件路径（相对于项目根目录）
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "strategy.yaml"


class StrategyConfig(BaseModel):
    """
    顶层策略配置容器（Pydantic 模型）。
    持有所有子模块的配置对象，后续 Ticket 的配置（如 GridConfig）也会加入此处。
    """
    regime_filter: RegimeFilterConfig = RegimeFilterConfig()


def load_strategy_config(path: Optional[str | Path] = None) -> StrategyConfig:
    """
    从 YAML 文件加载策略参数并返回 StrategyConfig 实例。

    Args:
        path: YAML 文件路径。默认为 config/strategy.yaml。

    Returns:
        StrategyConfig 实例，包含所有子模块的配置对象。

    Raises:
        FileNotFoundError: 配置文件不存在时抛出。
        ValidationError:   参数值不合法时由 Pydantic 自动抛出。
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"策略配置文件不存在: {config_path}\n"
            f"请确认 config/strategy.yaml 文件存在。"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    logger.info("策略配置文件加载成功: %s", config_path)

    # Pydantic 会自动校验所有字段，非法值直接抛出 ValidationError
    config = StrategyConfig.model_validate(raw)

    logger.info("策略配置解析完成: %s", config.model_dump())
    return config
