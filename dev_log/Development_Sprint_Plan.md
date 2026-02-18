# 开发冲刺计划 (Development Sprint Plan) - Binance BTC 自适应网格策略 (v2.0)

本计划将策略开发拆分为 6 个连续的功能 Ticket，以及辅助的调试工具。本版本已根据最新策略调整进行同步。

---

## ✅ Ticket-001: 基础架构与行情数据模块 (Project Foundation)
- **状态:** 已完成
- **核心功能:**
    - 初始化 Python 环境，实现 `ExchangeClient` 封装 CCXT。
    - 支持 `testnet` 和 `live` 模式切换。
    - 实现 K 线抓取 (`fetch_ohlcv`) 及增量更新逻辑。

---

## ✅ Ticket-002: 市场状态过滤器算子 (Regime Filter Implementation)
- **状态:** 已完成 (v2.0)
- **核心功能:**
    - 实现 `RegimeFilter` 类，计算 ADX, ATR, BB, Hurst。
    - **v2.0 实现:** 
        - 引入 `Hurst < 0.5 & ADX < 25` 作为震荡入场条件。
        - 引入 `Hurst > 0.6 & ADX > 30` 作为趋势熔断条件。
        - 支持分级响应逻辑输出（Hurst 不同区间对应的仓位建议）。
        - 支持 `Hurst < 0.35` 触发利润转现货建议。

---

## ✅ Ticket-ADH-001: 策略可视化调试面板 (Visualization Dashboard)
- **状态:** 已完成 (v2.0)
- **核心功能:**
    - 支持展示 MA200/MA120/MA60 基准线（后端逻辑已备）。
    - 侧边栏增加仓位利用率（Hurst 分级结果）显示。
    - 增加“利润转现货 (PROFIT TO BTC)”状态指示。
    - 全参数实时调参支持。

---

## 🔄 Ticket-003: 自适应网格核心逻辑 (Adaptive Grid Engine)
- **依赖:** Ticket-002
- **状态:** 已实现计算算子 (`src/grid_engine.py`)
- **主要需求 (v2.0):**
    - **基准线分配:** 引入 MA200/MA120 判断。
        - 价格 > MA：80% 资金利用率。
        - 价格 < MA：30% 资金利用率。
    - **动态区间:** 使用 `Current_Price ± 3 * ATR` 锁定网格上下界。
    - **中轴线:** 以 MA20 为中轴进行动态偏移。
    - **单向保护:** 
        - 触碰 BB 2.5σ 上轨 且 MA 偏离度 > 5% → **只卖不买**。
- **验收标准 (AC):**
    - 能根据 MA200 位置自动调整生成的 Total Investment 规模。
    - 高位超买状态下，生成的订单列表仅包含 Sell Orders。

---

## 🛠️ Ticket-004: 回测系统与性能评估 (Backtesting Framework)
- **依赖:** Ticket-003
- **需求:**
    - 模拟撮合逻辑，支持手续费。
    - **核心验证:** 验证“利润转现货”逻辑（Hurst < 0.35 时）。
    - 统计多空背景下的资金利用率切换效果。

---

## 🛠️ Ticket-005: 交易执行引擎 (Live Execution Layer)
- **依赖:** Ticket-004
- **需求:**
    - 封装 Binance 订单接口。
    - 实现订单同步与轮询机制。
    - 支持 Testnet 运行 24/7。

---

## 🛠️ Ticket-006: 风险控制与熔断监控 (Risk Control & Fusing)
- **依赖:** Ticket-005
- **需求 (v2.0):**
    - **OI 监控:** 实时获取 Binance Open Interest，1h 增长 > 12% 立即熔断撤单。
    - **资金费率:** Funding Rate 异常时暂停操作。
    - **CVD 背离:** 检测累计成交量差与价格的背离信号。
    - **Telegram 告警:** 异常触发第一时间推送。
- **验收标准 (AC):**
    - 模拟 OI 异常激增，程序需在 5 秒内完成撤单并进入 PAUSE 状态。
