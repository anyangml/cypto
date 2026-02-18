
import time
import pandas as pd
from typing import List, Dict, Optional
from src.config import Config
from src.strategy_config import load_strategy_config
from src.regime_filter import RegimeFilter, MarketRegime
from src.grid_engine import GridEngine
from src.risk_control import RiskManager, RiskStatus
from src.exchange import ExchangeClient
from src.database import DatabaseHandler
from src.logger import setup_logger

logger = setup_logger("execution")

class GridStrategyExecutor:
    """
    实盘/Testnet 策略执行器。
    
    负责调度：
    1. 获取市场数据 (Data)
    2. 计算市场状态 (Regime)
    3. 生成网格计划 (Grid)
    4. 执行订单同步 (Execution)
    5. 风险控制中断 (Risk Control)
    """
    
    def __init__(self):
        self.cfg = Config()
        self.strategy_cfg = load_strategy_config()
        
        # 初始化组件
        self.exchange = ExchangeClient(self.cfg)
        self.rf = RegimeFilter(self.strategy_cfg.regime_filter, self.strategy_cfg.position_sizing.tiers)
        self.ge = GridEngine(self.strategy_cfg.grid_engine)
        self.rm = RiskManager(self.strategy_cfg.risk_control, self.strategy_cfg.position_sizing.fuse_conditions)
        self.db = DatabaseHandler() # 默认路径 data/trades.db
        
        self.symbol = self.cfg.symbol
        self.timeframe = self.cfg.timeframe
        
        self.last_sync_trades_time = 0 # 初始化上次同步时间
        
        logger.info("GridStrategyExecutor 初始化完成")

    def run(self):
        """主循环"""
        logger.info("策略引擎启动...", "green")
        
        while True:
            try:
                self.step()
                
                # 等待下一个周期
                # 简单实现：每 60 秒运行一次，或者等待下一个 K 线收盘
                # 为了响应及时，建议 shorter interval，比如 30s
                wait_sec = 30
                logger.info(f"等待 {wait_sec} 秒后进入下一轮检测...")
                time.sleep(wait_sec)
                
            except KeyboardInterrupt:
                logger.warning("收到停止信号，正在退出...")
                break
            except Exception as e:
                logger.error("主循环发生未知错误: %s", e)
                time.sleep(10)

    def step(self):
        """单次执行逻辑"""
        logger.info("=== 开始新一轮策略计算 ===")
        
        # 1. 账户与数据同步
        balance = self.exchange.fetch_balance()
        # 估算总权益 (近似值，用于 Grid Engine 资金分配)
        # TODO: 获取真实标记价格计算
        # total_equity = balance.get("USDT", {}).get("total", 0) 
        
        df = self.exchange.fetch_ohlcv(since_days=self.strategy_cfg.regime_filter.hurst_period * 2 // 24 + 1)
        if df.empty:
            logger.warning("获取不到 K 线数据，跳过")
            return
            
        current_price = df['close'].iloc[-1]
        
        # 简单的权益估算
        usdt_balance = balance.get("USDT", {}).get("free", 0) + balance.get("USDT", {}).get("used", 0)
        btc_balance = balance.get("BTC", {}).get("free", 0) + balance.get("BTC", {}).get("used", 0)
        total_equity = usdt_balance + btc_balance * current_price
        
        # 1.5 风控检查 (Ticket-006)
        try:
            fr = self.exchange.fetch_funding_rate()
            oi = self.exchange.fetch_open_interest()
            risk = self.rm.check(current_oi=oi, current_funding_rate=fr)
            
            if risk.status in [RiskStatus.FUSE, RiskStatus.PAUSE]:
                logger.error(f"风控熔断触发: {risk.reason} | Status: {risk.status}")
                self.exchange.cancel_all_orders()
                return # 中止本轮
                
        except Exception as e:
            logger.warning("风控数据获取失败，跳过检查: %s", e)

        # 2. Regime Filter
        try:
            regime_res = self.rf.get_market_regime(df)
        except ValueError as e:
            logger.warning("Regime 计算失败(数据不足?): %s", e)
            return

        logger.info(f"当前 Regime: {regime_res}")
        
        # 3. 决策分支
        if regime_res.regime == MarketRegime.FUSE:
            logger.warning("触发熔断状态！正在撤销所有订单...")
            self.exchange.cancel_all_orders()
            # 可以在这里发送告警
            return

        if regime_res.regime == MarketRegime.TREND:
            logger.info("趋势状态，暂停网格开单 (只卖不买 logic if holding?)")
            # 策略选择：全面撤单观望，还是保留卖单？
            # PRD: "单边趋势，暂停新开网格"。通常意味着清空所有挂单防止逆势
            self.exchange.cancel_all_orders()
            return
            
        # Regime == RANGE
        if regime_res.position_ratio <= 0:
            logger.info("震荡但仓位建议为 0 (高 Hurst)，撤单观望")
            self.exchange.cancel_all_orders()
            return
            
        # 4. 生成网格计划
        grid_plan = self.ge.calculate_grid(
            df, 
            total_balance=total_equity,
            position_ratio=regime_res.position_ratio
        )
        
        logger.info(f"网格计划: Range=[{grid_plan['lower_boundary']}, {grid_plan['upper_boundary']}], Mid={grid_plan['mid_price']}")
        
        # 5. 执行订单同步 (Diffing)
        # 获取最新的 open orders 用于对比
        try:
            open_orders = self.exchange.fetch_open_orders()
        except Exception as e:
            logger.error("无法获取挂单，跳过同步: %s", e)
            return

        self._sync_orders(grid_plan, open_orders, current_price, usdt_balance, btc_balance)
        
        # 6. 每 5 分钟同步一次历史成交记录 (防止遗漏)
        now = time.time()
        if now - self.last_sync_trades_time > 300: # 300s = 5min
            self._sync_trades()
            self.last_sync_trades_time = now
            
        logger.info("---------------- Step End ----------------")

    def _sync_trades(self):
        """同步交易所成交记录到本地数据库"""
        try:
            # 获取最近一天的成交记录 (根据 API 限制调整)
            # symbol 必须传
            trades = self.exchange.client.fetch_my_trades(symbol=self.symbol, limit=50)
            if trades:
                self.db.record_trades(trades)
        except Exception as e:
            logger.warning(f"同步成交记录失败: {e}")

    def _sync_orders(self, plan: dict, open_orders: list, current_price: float, available_usdt: float, available_btc: float):
        """
        智能订单同步算法。
        比较现有挂单与计划挂单，进行最小化调整 (Diffing)。
        """
        # 建立当前挂单的映射表: price -> order_object
        # key 使用字符串保留精度，或者 round(price, 2)
        # 这里假设 stepSize=0.01
        def price_key(p): return round(float(p), 2)
        
        current_buys = {price_key(o['price']): o for o in open_orders if o['side'] == 'buy'}
        current_sells = {price_key(o['price']): o for o in open_orders if o['side'] == 'sell'}
        
        plan_buy_orders = plan.get('buy_orders', [])
        plan_sell_orders = plan.get('sell_orders', [])
        
        # 提取计划价格集合
        plan_buy_prices = {price_key(p['price']) for p in plan_buy_orders}
        plan_sell_prices = {price_key(p['price']) for p in plan_sell_orders}
        
        logger.info(f"Diffing: 当前买单{len(current_buys)} vs 计划{len(plan_buy_prices)} | 当前卖单{len(current_sells)} vs 计划{len(plan_sell_prices)}")
        
        # 1. 撤销不再需要的订单 (多余的，或价格偏离的)
        for p, order in current_buys.items():
            if p not in plan_buy_prices:
                logger.info(f"撤销多余买单: {p}")
                try: self.exchange.cancel_order(order['id'])
                except: pass

        for p, order in current_sells.items():
            if p not in plan_sell_prices:
                logger.info(f"撤销多余卖单: {p}")
                try: self.exchange.cancel_order(order['id'])
                except: pass
                
        # 2. 补挂缺失的订单
        # 计算单笔资金 (简单均分：可用余额 / 待挂单数)
        # 注意：这里需要更复杂的资金管理，防止余额不足
        # 暂时使用固定最小单量进行测试
        
        params = self.strategy_cfg.grid_engine
        buy_enabled = plan.get('buy_grid_enabled', True)
        sell_enabled = plan.get('sell_grid_enabled', True)
        
        # 最小下单量 (BTC)
        min_qty = 0.0004 
        
        if buy_enabled:
            for item in plan_buy_orders:
                p_key = price_key(item['price'])
                if p_key not in current_buys:
                    if float(item['price']) < current_price: # 再次确认价格
                        # 检查余额 (可选)
                        try:
                            self.exchange.create_limit_order("buy", float(item['price']), min_qty)
                            time.sleep(0.1) # 避免频率限制
                        except Exception as e:
                            logger.error(f"补挂买单失败 {item['price']}: {e}")

        if sell_enabled:
            for item in plan_sell_orders:
                p_key = price_key(item['price'])
                if p_key not in current_sells:
                    if float(item['price']) > current_price:
                        try:
                            self.exchange.create_limit_order("sell", float(item['price']), min_qty)
                            time.sleep(0.1)
                        except Exception as e:
                            logger.error(f"补挂卖单失败 {item['price']}: {e}")

    def _sync_trades(self):
        """同步交易所成交记录到本地数据库"""
        try:
            # 获取最近一天的成交记录 (根据 API 限制调整)
            # symbol 必须传
            trades = self.exchange.client.fetch_my_trades(symbol=self.symbol, limit=50)
            if trades:
                self.db.record_trades(trades)
        except Exception as e:
            logger.warning(f"同步成交记录失败: {e}")

if __name__ == "__main__":
    executor = GridStrategyExecutor()
    executor.run()
