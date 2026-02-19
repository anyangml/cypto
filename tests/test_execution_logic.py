
import pytest
from unittest.mock import MagicMock
from src.execution import GridStrategyExecutor

# ----------------------------------------------------------------------
# Mock Classes
# ----------------------------------------------------------------------

class MockExchange:
    def __init__(self):
        self.orders = []
        self.balance = {"USDT": 1000.0, "BTC": 0.1}
        self.symbol = "BTC/USDT"
    
    def fetch_balance(self):
        return self.balance
    
    def price_to_precision(self, symbol, price):
        return round(float(price), 2)
        
    def amount_to_precision(self, symbol, amount):
        return round(float(amount), 4)

    def create_limit_order(self, side, price, amount):
        self.orders.append({
            "id": f"ord_{len(self.orders)}",
            "side": side,
            "price": price,
            "amount": amount
        })
        return {"id": "mock_id"}

    def cancel_order(self, order_id):
        pass

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

class TestExecutionWithCapital:

    @pytest.fixture
    def executor(self):
        # 初始化 Executor 并注入 Mock Exchange
        executor = GridStrategyExecutor()
        executor.exchange = MagicMock()
        
        # 配置 Mock Exchange 的行为
        executor.exchange.fetch_balance.return_value = {"USDT": 10000.0, "BTC": 0.5}
        
        # 简单的精度处理 Mock
        def p_prec(sym, p): return round(float(p), 2)
        def a_prec(sym, a): return round(float(a), 4)
        
        executor.exchange.price_to_precision.side_effect = p_prec
        executor.exchange.amount_to_precision.side_effect = a_prec
        
        return executor

    def test_sync_orders_full_allocation(self, executor):
        """测试全量资金分配"""
        # 设置场景
        plan = {
            "buy_orders": [{"price": 10000}, {"price": 10100}],
            "sell_orders": [],
            "capital_utilization": 1.0,
            "buy_grid_enabled": True
        }
        open_orders = [] # 当前无挂单
        current_price = 10200 # 全部买单都有效
        
        # 执行
        executor._sync_orders(plan, open_orders, current_price)
        
        # 验证
        # 余额 10000 USDT，利用率 1.0，两个买单
        # 每个买单约 5000 USDT (考虑到 buffer 0.98 -> 4900)
        # Price 10000 -> qty 0.49
        
        calls = executor.exchange.create_limit_order.call_args_list
        assert len(calls) == 2
        
        # 检查第一笔调用的参数
        side, price, qty = calls[0][0]
        assert side == "buy"
        assert price in [10000.0, 10100.0]
        assert 0.48 < qty < 0.50 # 验证数量级

    def test_sync_orders_partial_utilization(self, executor):
        """测试部分资金利用率"""
        plan = {
            "buy_orders": [{"price": 9000}], 
            "sell_orders": [],
            "capital_utilization": 0.5, # 只用一半资金
            "buy_grid_enabled": True
        }
        open_orders = []
        
        # 余额 10000 -> 可用 5000 -> 单笔 4900
        executor._sync_orders(plan, open_orders, 10000)
        
        calls = executor.exchange.create_limit_order.call_args_list
        side, price, qty = calls[0][0]
        
        expected_qty = (10000 * 0.5 * 0.98) / 9000
        assert abs(qty - expected_qty) < 0.01

    def test_sync_orders_min_notional_check(self, executor):
        """测试最小名义价值检查"""
        # 余额很小
        executor.exchange.fetch_balance.return_value = {"USDT": 10.0, "BTC": 0.0}
        
        # 计划挂 10 个买单 -> 每个 1U -> 应该被拒绝或合并
        plan = {
            "buy_orders": [{"price": 100-i} for i in range(10)],
            "sell_orders": [],
            "capital_utilization": 1.0
        }
        
        executor._sync_orders(plan, [], 200)
        
        # 由于 budget ~9.8U，每个单 < 5U
        # 逻辑应该会自动削减订单数，只挂 1 个 (9.8U)
        calls = executor.exchange.create_limit_order.call_args_list
        assert len(calls) == 1
        
        # 且应该是价格最高的那个 (接近 100)
        price = calls[0][0][1]
        assert price == 100.0

