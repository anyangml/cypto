import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from src.logger import setup_logger

logger = setup_logger("database")

class DatabaseHandler:
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = Path(db_path)
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 创建 trades 表
        # id 是交易所返回的唯一成交ID，用于去重
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                price REAL,
                qty REAL,
                quote_qty REAL,
                fee REAL,
                fee_currency TEXT,
                timestamp INTEGER,
                datetime TEXT,
                order_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def record_trades(self, trades: List[Dict]):
        """
        批量插入交易记录 (自动忽略已存在的 ID)
        Args:
            trades: CCXT 格式的 trade 字典列表
        """
        if not trades:
            return

        conn = self._get_conn()
        try:
            count = 0
            for trade in trades:
                # 解析费用
                fee_cost = 0.0
                fee_curr = ""
                if trade.get('fee'):
                    fee_cost = float(trade['fee'].get('cost', 0.0))
                    fee_curr = trade['fee'].get('currency', '')

                conn.execute('''
                    INSERT OR IGNORE INTO trades 
                    (id, symbol, side, price, qty, quote_qty, fee, fee_currency, timestamp, datetime, order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(trade['id']),
                    str(trade['symbol']),
                    str(trade['side']).upper(),
                    float(trade['price']),
                    float(trade['amount']),
                    float(trade['cost']) if trade.get('cost') else float(trade['price']) * float(trade['amount']),
                    fee_cost,
                    fee_curr,
                    int(trade['timestamp']),
                    str(trade['datetime']),
                    str(trade['order'])
                ))
                count += conn.changes()
            
            conn.commit()
            if count > 0:
                logger.info(f"Recorded {count} new trades to database")
                
        except Exception as e:
            logger.error(f"Failed to record trades: {e}")
        finally:
            conn.close()

    def get_trades(self, limit: int = 100) -> List[Dict]:
        """获取最近的交易记录"""
        conn = self._get_conn()
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM trades ORDER BY timestamp DESC LIMIT {limit}", 
                conn
            )
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        """获取简单的统计数据"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), SUM(quote_qty) FROM trades")
            row = cursor.fetchone()
            total_trades = row[0] if row else 0
            total_volume = row[1] if row and row[1] else 0.0
            return {
                "total_trades": total_trades,
                "total_volume": round(total_volume, 2)
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_trades": 0, "total_volume": 0.0}
        finally:
            conn.close()
