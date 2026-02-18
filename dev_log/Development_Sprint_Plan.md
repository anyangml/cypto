# 开发冲刺计划 (Development Sprint Plan) - Binance BTC 自适应网格策略

本计划将策略开发拆分为 6 个连续的功能 Ticket，每个 Ticket 均建立在前一个 Ticket 完成的基础上。

---

## Ticket-001: 基础架构与行情数据模块 (Project Foundation)
- **依赖:** 无
- **需求:**
    - 初始化 Python 开发环境（CCXT, Pandas, Logging）。
    - 实现抽象的交易所连接层，支持通过配置文件读取 Binance API Key/Secret。
    - 实现行情抓取功能：能够从币安获取历史 K 线数据（用于回测）和实时 K 线（用于实盘）。
- **验收标准 (AC):**
    - 成功连接币安 API 并能打印当前账户余额。
    - 能够下载过去 30 天的 BTC/USDT 1h K 线数据并保存为 CSV。
    - 统一的日志系统能够记录 `INFO` 和 `ERROR` 级别信息。

---

## Ticket-002: 市场状态过滤器算子 (Regime Filter Implementation)
- **依赖:** Ticket-001
- **需求:**
    - 实现 `RegimeFilter` 类。
    - 集成逻辑：计算 ADX, ATR, Bollinger Bands 宽度和 Hurst 指数。
    - 实现核心判断函数 `get_market_regime()`：返回 `RANGE` (震荡) 或 `TREND` (趋势)。
- **验收标准 (AC):**
    - 输入一段历史 K 线数据，能够准确输出每个时间点的市场状态（0 为震荡，1 为趋势）。
    - 提供单元测试，验证在明显的单边行情和横盘行情下，过滤器输出符合预期。

---

## Ticket-003: 自适应网格核心逻辑 (Adaptive Grid Engine)
- **依赖:** Ticket-002
- **需求:**
    - 开发网格逻辑算子：根据当前价格、ATR 波动率自动计算网格步长和层级。
    - 实现“中轴线动态调整”：网格基准价格随 EMA 缓慢漂移。
    - 定义订单管理逻辑：挂单价格计算、平衡买卖单对（Buy-Sell Pairs）。
- **验收标准 (AC):**
    - 逻辑层能够计算出具体的挂单价格列表（List of Limit Prices）。
    - 当 ATR 增大时，计算出的网格间距必须相应变宽。

---

## Ticket-004: 回测系统与可视化分析 (Backtesting Framework)
- **依赖:** Ticket-003, Ticket-002
- **需求:**
    - 构建一个基于历史数据的回测模拟器。
    - 模拟撮合逻辑：根据历史 K 线的高低点判断网格订单是否成交。
    - 性能评估：计算累计收益、最大回撤、夏普比率。
    - 可视化：使用 Matplotlib 或 Plotly 绘制收益曲线，并标注 Regime Filter 的切换点。
- **验收标准 (AC):**
    - 回测报告需输出最终净值、胜率和最大回撤百分比。
    - 图表中清晰显示在何处策略因“检测到趋势”而停止了网格操作。

---

## Ticket-005: 交易执行引擎 (Live Execution Layer)
- **依赖:** Ticket-004, Ticket-001
- **需求:**
    - 将网格算子对接真实的 Binance 订单接口（限价单）。
    - 实现订单跟踪逻辑：轮询或通过 WebSocket 监听订单成交状态。
    - 实现“挂卖补买/挂买补卖”的循环逻辑。
- **验收标准 (AC):**
    - 在 Binance Testnet (测试网) 上成功运行 24 小时，无程序崩溃。
    - 能够自动处理部分成交订单，并在订单成交后立即挂出对冲平仓单。

---

## Ticket-006: 风险控制与告警监控 (Risk Control & Monitoring)
- **依赖:** Ticket-005
- **需求:**
    - 实现全局止损触发器：当资产跌破预设阈值时，撤销所有网格单并市价平仓。
    - 实现趋势强保护：若 Regime Filter 持续输出趋势信号，自动暂停所有新开网格。
    - 接入 Telegram/Discord API：实时推送交易状态、每日盈亏报告及异常告警。
- **验收标准 (AC):**
    - 手动模拟一个大跌场景，策略必须能触发止损并发送告警通知。
    - 每天定时发送一次包含当前持仓和总资产的 Daily Report。
