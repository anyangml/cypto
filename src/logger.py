"""
logger.py - 统一日志模块

提供项目全局使用的 logger 实例，支持控制台和文件分层输出。
"""

import logging
import os
from datetime import datetime


def setup_logger(
    name: str = "grid_strategy", 
    log_dir: str = "logs", 
    level: str = "INFO"
) -> logging.Logger:
    """
    初始化并返回一个 Logger 实例。

    Args:
        name:    Logger 名称。
        log_dir: 日志目录。
        level:   控制台输出级别 (DEBUG, INFO, WARNING, ERROR)。
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    
    # 避免在多次调用时重复添加 Handler
    if logger.handlers:
        return logger

    # 基础级别设为 DEBUG，允许 Handler 进一步过滤
    logger.setLevel(logging.DEBUG)

    # 映射字符串级别到 logging 常量
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # --- 日志格式 ---
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- 控制台 Handler (可动态配置) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)

    # --- 文件 Handler (强制保留 DEBUG 级别，用于死后验尸) ---
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 默认预置一个 INFO 级别的 logger
logger = setup_logger()
