
import pandas as pd
import numpy as np
from typing import List, Dict
from src.config import Config, TradingMode
from src.strategy_config import load_strategy_config
from src.regime_filter import RegimeFilter, MarketRegime
from src.grid_engine import GridEngine
from src.logger import setup_logger

logger = setup_logger("backtest")

class BacktestEngine:
    def __init__(self, strategy_cfg=None):
        self.cfg = Config()
        self.strategy_cfg = strategy_cfg or load_strategy_config()
        self.rf = RegimeFilter(self.strategy_cfg.regime_filter, self.strategy_cfg.position_sizing.tiers)
        self.ge = GridEngine(self.strategy_cfg.grid_engine)
        
        # 初始资金
        self.initial_usdt = 10000.0
        self.initial_btc = 0.0
        
        self.balance_usdt = self.initial_usdt
        self.balance_btc = self.initial_btc
        
        self.orders: List[Dict] = [] # Active orders
        self.trades: List[Dict] = []
        
    def run(self, df: pd.DataFrame) -> Dict:
        """
        运行回测
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            
        Returns:
            Dict: {
                "equity_curve": [{"time": ts, "value": val}, ...],
                "trades": [{"time": ts, "side": "buy", "price": p, "qty": q, "profit": ...}, ...],
                "metrics": {"total_return": ..., "max_drawdown": ...}
            }
        """
        self.df = df
        # 重置状态
        self.balance_usdt = self.initial_usdt
        self.balance_btc = self.initial_btc
        self.orders = []
        self.trades = []
        
        # 预计算 Regime
        logger.info("正在计算 Regime 信号...")
        regimes = self.rf.scan_dataframe(self.df)
        
        equity_curve = []
        
        # 模拟逐 K 线撮合
        # 确保数据长度足够 GridEngine 计算 MA200
        start_idx = max(self.ge.config.ma_long_period, 100)
        
        for i in range(len(self.df)):
            if i < start_idx: continue 
            
            # 当前切片
            current_idx = self.df.index[i]
            current_bar = self.df.iloc[i]
            past_df = self.df.iloc[:i+1]
            
            # 1. 撮合上一时刻的挂单
            self._match_orders(current_bar)
            
            # 2. 获取当前状态
            regime = regimes.iloc[i]
            
            # 3. 如果是 RANGE 且非 FUSE，生成新网格
            # 注意：实际策略不会每根 K 线都重置网格，通常是偏离中轴一定程度才重置
            # 这里简化为：如果没有挂单，或者 Regime 发生变化，则重置
            
            if regime == "RANGE":
                # 获取实时 Regime 结果以读取 position_ratio
                regime_res = self.rf.get_market_regime(past_df)
                
                # 只有当仓位 > 0 时才开网格
                if regime_res.position_ratio > 0 and not self.orders:
                    # 全撤全挂逻辑
                    self.orders = [] 
                    
                    grid_plan = self.ge.calculate_grid(
                        past_df, 
                        total_balance=self.total_value(current_bar['close']),
                        position_ratio=regime_res.position_ratio
                    )
                    
                    # 转换 plan 为订单
                    for bo in grid_plan['buy_orders']:
                        # 简单资金分配：总买单资金 / 订单数
                        # 实际应根据 grid_engine 逻辑
                        amount_usdt = (self.balance_usdt * 0.9) / len(grid_plan['buy_orders'])
                        qty = amount_usdt / bo['price']
                        if qty > 0.0001:
                            self.orders.append({
                                'side': 'buy',
                                'price': bo['price'],
                                'qty': qty,
                                'id': f"buy_{i}_{bo['price']}"
                            })

                    for so in grid_plan['sell_orders']:
                        # 简单分币：总持仓 / 订单数
                        if self.balance_btc > 0.0001:
                            qty = self.balance_btc / len(grid_plan['sell_orders'])
                            if qty > 0.0001:
                                self.orders.append({
                                    'side': 'sell',
                                    'price': so['price'],
                                    'qty': qty,
                                    'id': f"sell_{i}_{so['price']}"
                                })
            
            elif regime == "TREND" or regime == "FUSE":
                # 趋势或熔断，清空挂单
                self.orders = []
            
            # 记录权益
            val = self.total_value(current_bar['close'])
            # 简化数据量，只记录关键点
            equity_curve.append({
                'time': int(current_idx.timestamp()),
                'value': round(val, 2),
                'regime': regime
            })

        logger.info("回测结束")
        final_val = equity_curve[-1]['value'] if equity_curve else self.initial_usdt
        ret = (final_val - self.initial_usdt) / self.initial_usdt
        
        # 计算最大回撤
        max_val = 0
        max_dd = 0
        for p in equity_curve:
            if p['value'] > max_val:
                max_val = p['value']
            dd = (max_val - p['value']) / max_val
            if dd > max_dd:
                max_dd = dd

        return {
            "equity_curve": equity_curve,
            "trades": self.trades,
            "metrics": {
                "total_return_pct": round(ret * 100, 2),
                "final_value": round(final_val, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "trade_count": len(self.trades)
            }
        }

    def _match_orders(self, bar):
        # 简单的撮合逻辑：
        # Buy: if Low < Price
        # Sell: if High > Price
        # 优先成交：假设 High/Low 都能覆盖
        
        remaining_orders = []
        current_time = int(bar.name.timestamp())
        
        for order in self.orders:
            executed = False
            if order['side'] == 'buy':
                if bar['low'] < order['price']:
                    # 成交
                    cost = order['price'] * order['qty'] * (1 + 0.001) # 费率千1
                    if self.balance_usdt >= cost:
                        self.balance_usdt -= cost
                        self.balance_btc += order['qty']
                        executed = True
                        self.trades.append({
                            "time": current_time, # 使用 K 线时间戳作为成交时间
                            "side": "buy",
                            "price": order['price'],
                            "qty": order['qty']
                        })
            
            elif order['side'] == 'sell':
                if bar['high'] > order['price']:
                    # 成交
                    revenue = order['price'] * order['qty'] * (1 - 0.001)
                    if self.balance_btc >= order['qty']:
                        self.balance_btc -= order['qty']
                        self.balance_usdt += revenue
                        executed = True
                        self.trades.append({
                            "time": current_time,
                            "side": "sell",
                            "price": order['price'],
                            "qty": order['qty']
                        })
            
            if not executed:
                remaining_orders.append(order)
        
        self.orders = remaining_orders

    def total_value(self, price):
        return self.balance_usdt + self.balance_btc * price

if __name__ == "__main__":
    bt = BacktestEngine()
    bt.load_data_from_exchange(days=90)
    bt.run()
