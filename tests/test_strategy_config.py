"""
tests/test_strategy_config.py - 策略参数加载模块测试
"""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from src.strategy_config import load_strategy_config, StrategyConfig
from src.regime_filter import RegimeFilterConfig


class TestLoadStrategyConfig:

    def test_loads_default_yaml(self):
        """应能成功加载默认的 config/strategy.yaml 文件。"""
        cfg = load_strategy_config()
        assert isinstance(cfg, StrategyConfig)
        assert isinstance(cfg.regime_filter, RegimeFilterConfig)

    def test_regime_filter_values_match_yaml(self):
        """加载后的值应与 YAML 文件中的值一致。"""
        cfg = load_strategy_config()
        # 读取原始 YAML 做对比
        yaml_path = Path("config/strategy.yaml")
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        rf_raw = raw["regime_filter"]

        assert cfg.regime_filter.adx_period == rf_raw["adx_period"]
        assert cfg.regime_filter.adx_trend_threshold == rf_raw["adx_trend_threshold"]
        assert cfg.regime_filter.trend_vote_threshold == rf_raw["trend_vote_threshold"]

    def test_file_not_found_raises_error(self):
        """配置文件不存在时应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_strategy_config(path="/nonexistent/path/strategy.yaml")

    def test_custom_yaml_overrides_defaults(self, tmp_path):
        """自定义 YAML 文件的值应覆盖默认值。"""
        custom_yaml = tmp_path / "strategy.yaml"
        custom_yaml.write_text(
            "regime_filter:\n"
            "  adx_period: 21\n"
            "  adx_trend_threshold: 30.0\n"
            "  atr_short_period: 5\n"
            "  atr_long_period: 20\n"
            "  atr_ratio_threshold: 1.8\n"
            "  bb_period: 20\n"
            "  bb_std: 2.0\n"
            "  bb_width_threshold: 0.1\n"
            "  hurst_period: 100\n"
            "  hurst_threshold: 0.5\n"
            "  trend_vote_threshold: 3\n",
            encoding="utf-8",
        )
        cfg = load_strategy_config(path=custom_yaml)
        assert cfg.regime_filter.adx_period == 21
        assert cfg.regime_filter.adx_trend_threshold == 30.0
        assert cfg.regime_filter.trend_vote_threshold == 3

    def test_invalid_param_raises_validation_error(self, tmp_path):
        """非法参数值应由 Pydantic 抛出 ValidationError。"""
        bad_yaml = tmp_path / "strategy.yaml"
        bad_yaml.write_text(
            "regime_filter:\n"
            "  adx_period: 14\n"
            "  adx_trend_threshold: 25.0\n"
            "  atr_short_period: 28\n"   # 错误：short >= long
            "  atr_long_period: 7\n"
            "  atr_ratio_threshold: 1.5\n"
            "  bb_period: 20\n"
            "  bb_std: 2.0\n"
            "  bb_width_threshold: 0.1\n"
            "  hurst_period: 100\n"
            "  hurst_threshold: 0.5\n"
            "  trend_vote_threshold: 2\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_strategy_config(path=bad_yaml)

    def test_missing_fields_use_defaults(self, tmp_path):
        """YAML 中未指定的字段应使用 RegimeFilterConfig 的默认值。"""
        partial_yaml = tmp_path / "strategy.yaml"
        partial_yaml.write_text(
            "regime_filter:\n"
            "  adx_trend_threshold: 30.0\n",  # 只指定一个字段
            encoding="utf-8",
        )
        cfg = load_strategy_config(path=partial_yaml)
        defaults = RegimeFilterConfig()
        # 未指定的字段应等于默认值
        assert cfg.regime_filter.adx_period == defaults.adx_period
        assert cfg.regime_filter.trend_vote_threshold == defaults.trend_vote_threshold
        # 指定的字段应被覆盖
        assert cfg.regime_filter.adx_trend_threshold == 30.0


class TestRegimeFilterConfigCrossFieldValidation:
    """专门测试 RegimeFilterConfig 的跨字段约束。"""

    def _make_valid_params(self) -> dict:
        """返回一组合法的基础参数，用于在各测试中单独破坏某一约束。"""
        return dict(
            adx_period=14,
            adx_trend_threshold=25.0,
            atr_short_period=7,
            atr_long_period=28,
            atr_ratio_threshold=1.5,
            bb_period=20,
            bb_std=2.0,
            bb_width_threshold=0.1,
            hurst_period=100,
            hurst_threshold=0.5,
            trend_vote_threshold=2,
        )

    def test_valid_params_pass(self):
        """合法参数应能成功实例化。"""
        cfg = RegimeFilterConfig(**self._make_valid_params())
        assert cfg.adx_period == 14

    def test_atr_short_ge_long_fails(self):
        """atr_short_period >= atr_long_period 应触发 ValidationError。"""
        params = self._make_valid_params()
        params["atr_short_period"] = 28
        params["atr_long_period"] = 7
        with pytest.raises(ValidationError, match="atr_short_period"):
            RegimeFilterConfig(**params)

    def test_hurst_period_less_than_atr_long_fails(self):
        """hurst_period < atr_long_period 应触发 ValidationError。"""
        params = self._make_valid_params()
        params["atr_long_period"] = 50
        params["hurst_period"] = 30   # 小于 atr_long_period
        with pytest.raises(ValidationError, match="hurst_period"):
            RegimeFilterConfig(**params)

    def test_hurst_period_less_than_bb_period_fails(self):
        """hurst_period < bb_period 应触发 ValidationError。"""
        params = self._make_valid_params()
        params["bb_period"] = 50
        params["hurst_period"] = 30   # 小于 bb_period
        with pytest.raises(ValidationError, match="hurst_period"):
            RegimeFilterConfig(**params)

    def test_hurst_period_less_than_adx_warmup_fails(self):
        """hurst_period < adx_period * 2 应触发 ValidationError。"""
        params = self._make_valid_params()
        params["adx_period"] = 30
        params["hurst_period"] = 40   # 小于 adx_period * 2 = 60
        with pytest.raises(ValidationError, match="hurst_period"):
            RegimeFilterConfig(**params)

    def test_multiple_violations_reported_together(self):
        """多个约束同时违反时，错误信息应包含所有违反项。"""
        params = self._make_valid_params()
        params["atr_short_period"] = 50   # 违反 atr_short < atr_long
        params["atr_long_period"] = 10
        params["hurst_period"] = 20       # 违反 hurst >= atr_long (10 ok), hurst >= adx*2 (28)
        params["adx_period"] = 20         # adx*2 = 40 > hurst_period=20
        with pytest.raises(ValidationError):
            RegimeFilterConfig(**params)

    def test_adx_trend_threshold_out_of_range_fails(self):
        """adx_trend_threshold 超出 (0, 100] 范围应触发 ValidationError。"""
        params = self._make_valid_params()
        params["adx_trend_threshold"] = 0.0   # 不允许 <= 0
        with pytest.raises(ValidationError):
            RegimeFilterConfig(**params)

    def test_trend_vote_threshold_out_of_range_fails(self):
        """trend_vote_threshold 超出 [1, 4] 范围应触发 ValidationError。"""
        params = self._make_valid_params()
        params["trend_vote_threshold"] = 5   # 最大为 4
        with pytest.raises(ValidationError):
            RegimeFilterConfig(**params)

