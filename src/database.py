"""
database.py - 本地 SQLite 持久化层

负责将交易所成交记录持久化到本地数据库，支持去重插入和统计查询。
"""

import sqlite3
from pathlib import Path
from typing import List, Dict

from src.logger import setup_logger

logger = setup_logger("database")


class DatabaseHandler:
    """
    SQLite 数据库处理器。

    每次操作均创建独立连接，天然支持多线程访问（FastAPI + 策略进程）。
    """

    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """创建并返回一个新的数据库连接（调用方负责关闭）。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 支持按列名访问
        return conn

    def _init_db(self):
        """初始化数据库表结构（幂等操作）。"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id           TEXT PRIMARY KEY,
                    symbol       TEXT NOT NULL,
                    side         TEXT NOT NULL,
                    price        REAL NOT NULL,
                    qty          REAL NOT NULL,
                    quote_qty    REAL NOT NULL,
                    fee          REAL NOT NULL DEFAULT 0.0,
                    fee_currency TEXT NOT NULL DEFAULT '',
                    timestamp    INTEGER NOT NULL,
                    datetime     TEXT NOT NULL,
                    order_id     TEXT NOT NULL DEFAULT ''
                )
            """)
            # 常用查询索引
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC)"
            )
        logger.info("Database initialized at %s", self.db_path)

    def record_trades(self, trades: List[Dict]) -> int:
        """
        批量插入成交记录，自动忽略已存在的 ID（幂等）。

        Args:
            trades: CCXT 格式的 trade 字典列表。

        Returns:
            实际新增的记录数。
        """
        if not trades:
            return 0

        rows = []
        for trade in trades:
            fee = trade.get("fee") or {}
            rows.append((
                str(trade["id"]),
                str(trade["symbol"]),
                str(trade["side"]).upper(),
                float(trade["price"]),
                float(trade["amount"]),
                float(trade.get("cost") or float(trade["price"]) * float(trade["amount"])),
                float(fee.get("cost") or 0.0),
                str(fee.get("currency") or ""),
                int(trade["timestamp"]),
                str(trade["datetime"]),
                str(trade.get("order") or ""),
            ))

        inserted = 0
        try:
            with self._get_conn() as conn:
                for row in rows:
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO trades
                        (id, symbol, side, price, qty, quote_qty,
                         fee, fee_currency, timestamp, datetime, order_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        row,
                    )
                    inserted += cursor.rowcount
        except sqlite3.Error as e:
            logger.error("Failed to record trades: %s", e)
            return 0

        if inserted > 0:
            logger.info("Recorded %d new trade(s) to database", inserted)
        return inserted

    def get_trades(self, limit: int = 100) -> List[Dict]:
        """
        获取最近的成交记录。

        Args:
            limit: 返回条数上限，最大 500。
        """
        # 防止 limit 被滥用（即使类型是 int，也做上限保护）
        limit = max(1, min(int(limit), 500))
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error("Failed to fetch trades: %s", e)
            return []

    def get_stats(self) -> Dict:
        """获取汇总统计数据。"""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(quote_qty), 0.0) FROM trades"
                ).fetchone()
                return {
                    "total_trades": row[0],
                    "total_volume": round(row[1], 2),
                }
        except sqlite3.Error as e:
            logger.error("Failed to get stats: %s", e)
            return {"total_trades": 0, "total_volume": 0.0}
