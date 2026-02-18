
import time
from collections import deque
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

from src.strategy_config import RiskControlConfig, FuseConditions
from src.logger import setup_logger

logger = setup_logger("risk_control")

class RiskStatus(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING" # 仅告警，不停车
    PAUSE = "PAUSE"     # 暂停开新仓
    FUSE = "FUSE"       # 熔断，全撤单

class RiskAssessment(BaseModel):
    status: RiskStatus
    reason: str
    oi_change_1h: float = 0.0
    funding_rate: float = 0.0

class RiskManager:
    def __init__(self, config: RiskControlConfig, fuse_config: FuseConditions):
        self.config = config
        self.fuse_config = fuse_config
        
        # OI 历史记录: [(timestamp, oi_amount), ...]
        # 只需要保留过去 1 小时的数据
        self._oi_history: deque = deque() 
        self._last_check_time = 0

    def check(self, current_oi: float, current_funding_rate: float) -> RiskAssessment:
        """
        执行所有风控检查。
        
        Args:
            current_oi: 当前持仓量 (amount)
            current_funding_rate: 当前资金费率 (e.g., 0.0001)
            
        Returns:
            RiskAssessment 结果
        """
        now = time.time()
        
        # 1. 更新 OI 历史
        self._update_oi_history(now, current_oi)
        
        # 2. 检查资金费率
        fr_check = self._check_funding_rate(current_funding_rate)
        if fr_check:
            return fr_check
            
        # 3. 检查 OI 突变
        oi_check = self._check_oi_change(current_oi)
        if oi_check:
            return oi_check
            
        return RiskAssessment(status=RiskStatus.NORMAL, reason="All checks passed")

    def _update_oi_history(self, timestamp: float, oi: float):
        """维护 1 小时长的 OI 历史队列"""
        if oi <= 0: return # 忽略无效值
        
        self._oi_history.append((timestamp, oi))
        
        # 移除超过 1 小时的数据，但至少保留一个最旧的作为基准
        # 实际上我们需要 "1小时前" 的数据，所以保留稍微多一点，比如 65 分钟
        retention = 3600 + 300 
        while len(self._oi_history) > 1 and (timestamp - self._oi_history[0][0] > retention):
            self._oi_history.popleft()

    def _check_funding_rate(self, fr: float) -> Optional[RiskAssessment]:
        """资金费率检查"""
        if fr > self.config.funding_rate_max:
            return RiskAssessment(
                status=RiskStatus.PAUSE,
                reason=f"Funding Rate too high ({fr:.5f} > {self.config.funding_rate_max})",
                funding_rate=fr
            )
        if fr < self.config.funding_rate_min:
            return RiskAssessment(
                status=RiskStatus.PAUSE,
                reason=f"Funding Rate too low ({fr:.5f} < {self.config.funding_rate_min})",
                funding_rate=fr
            )
        return None

    def _check_oi_change(self, current_oi: float) -> Optional[RiskAssessment]:
        """OI 突变检查"""
        if not self._oi_history:
            return None
            
        # 寻找大约 1 小时前的数据点
        # 遍历队列，找到第一个 timestamp >= now - 3600
        now = self._oi_history[-1][0]
        one_hour_ago = now - 3600
        
        # 找到最接近 1 小时前的数据 (且在 1h ± 5min 范围内)
        base_oi = None
        for ts, val in self._oi_history:
            if ts >= one_hour_ago:
                base_oi = val
                break
        
        if base_oi is None:
            # 历史数据不足 1 小时，暂无法判断，或者使用由于是最早的一条
            base_oi = self._oi_history[0][1]
            # 如果数据太新（比如程序刚启动几分钟），差值没有统计意义，跳过
            if now - self._oi_history[0][0] < 600: # 至少 10 分钟
                return None

        if base_oi <= 0: return None

        pct_change = (current_oi - base_oi) / base_oi
        
        if abs(pct_change) > self.fuse_config.oi_change_1h_threshold:
            # 剧烈变化 -> 熔断
            return RiskAssessment(
                status=RiskStatus.FUSE,
                reason=f"OI Surge detected: {pct_change:.2%} in 1h",
                oi_change_1h=pct_change
            )
            
        return None
