# Binance BTC Grid Strategy (Regime-Adaptive)

这是一个基于市场状态（Market Regime）自适应的比特币网格交易机器人。它利用 Hurst 指数、ADX 和 ATR 等指标判断市场处于「震荡」还是「趋势」状态，并在震荡市中自动开启网格套利，在趋势市中自动止盈或空仓避险。

## 🚀 快速启动

### 1. 环境准备

确保已安装 Python 3.10+。

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置文件

复制 `.env.example` 到 `.env` 并填入您的 Binance API Key：

```bash
cp .env.example .env
```

**关键配置项 (.env)**：

```ini
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
# 实盘交易模式 (real) 或 模拟盘 (testnet)
TRADING_MODE=testnet
# 防止 Broker ID 抽水（默认开启）
BROKER_PREFIX=MyGrid
```

### 3. 启动 Dashboard

启动 Web 界面进行回测和监控：

```bash
kill -9 $(lsof -t -i:8000)
python dashboard/run.py
```

访问浏览器：`http://localhost:8000`

---

## 📊 功能说明

### 1. 回测系统 (Backtest)

回测引擎已经过深度优化，特征如下：
*   **真实模拟**：模拟 50% USDT + 50% BTC 初始持仓，从第一根 K 线即可双向开单。
*   **路径撮合**：动态模拟日内路径（阳线 `O->L->H->C`，阴线 `O->H->L->C`），最大限度还原真实成交顺序。
*   **资金利用率**：严格遵守 Regime Filter 输出的仓位比例（如 0.5 或 0.8），动态调整下单量。
*   **动态重置**：仅当价格偏离网格中轴 >1.5% 时才重置网格，大幅减少手续费。

**操作步骤**：
1.  打开 Dashboard。
2.  在右侧面板选择 "Backtest" 模式。
3.  设置回测天数（如 30 天）和初始资金（默认 10000 U）。
4.  点击 "Run" 按钮，查看资金曲线和交易明细。

### 2. 实盘引擎 (Live Execution)

实盘引擎位在 `src/execution.py`，它负责：
*   **状态同步**：每 30 秒同步一次账户余额和 K 线数据。
*   **Regime 识别**：实时计算 Hurst/ADX，判断当前是否适合开网格。
*   **订单管理**：
    *   **Diffing 算法**：智能对比当前挂单与计划单，只撤销/补挂必要的订单。
    *   **动态资金分配**：根据当前账户余额和 Regime 仓位比例，动态计算每格挂单金额。
    *   **精度保护**：自动处理交易所 Price/Amount 精度，并自动合并小于 5 USDT 的微小订单。

**启动命令**：

```bash
# 建议使用 screen 或 tmux 后台运行
python src/execution.py
```

**安全特性**：
*   **Anti-Broker-ID**：所有订单均强制附带自定义 `newClientOrderId`，防止 CCXT 库或中间商偷偷植入 Broker ID 抽取返佣。
*   **API 异常处理**：内置此 `adjustForTimeDifference` 和 `recvWindow=10000`，防止网络抖动导致的 API 拒绝。

---

## ⚙️ 策略配置详解

策略核心参数位于 `config/strategy.yaml`。

### Regime Filter (市场状态过滤)
```yaml
regime_filter:
  hurst_period: 100       # Hurst 指数计算周期
  hurust_threshold: 0.5   # 判定震荡的阈值 (<0.5 为均值回归)
  adx_trend_threshold: 25 # ADX > 25 视为趋势
```

### Grid Engine (网格引擎)
```yaml
grid_engine:
  grid_levels: 20         # 网格层数
  min_order_qty: 0.0004   # 最小下单量 (BTC)
  grid_reset_deviation: 0.015 # 价格偏离 1.5% 重置网格
```

---

## ❓ 常见问题 (FAQ)

**Q: 为什么回测显示有交易，但实盘不挂单？**
A: 请检查日志 (`logs/execution.log`)。
   1. **API Key 权限**：确保已开启 Spot Trading 权限。
   2. **余额不足**：实盘需要同时持有 USDT 和 BTC 才能双向挂单。如果只有 USDT，只会挂买单。
   3. **最小名义价值**：Binance 要求单笔订单 > 5 USDT。如果您的本金太少（例如 100 U 分 20 格），每格只有 5 U，很容易触发 `MIN_NOTIONAL` 错误导致挂单失败。建议每格资金 > 10 U。

**Q: 如何配置返佣给自己的 Broker 账号？**
A: 如果您有 Binance Broker ID（假设 API 返佣账号为 A，正在跑策略的账号为 B），请在 B 的 `.env` 文件中设置：
   `BROKER_PREFIX=x-您的BrokerID`
   这样 B 账号产生的手续费会返还给 A。

**Q: 报错 `Timestamp for this request is outside of the recvWindow`？**
A: 代码已默认开启 `adjustForTimeDifference`。如果依然报错，请检查服务器系统时间是否已同步 (`ntpdate -u pool.ntp.org`)。

---

**⚠️ 风险提示**：
加密货币市场波动剧烈。网格策略在单边暴跌行情中可能会遭受浮动亏损（满仓被套）。请务必合理设置止损或使用闲钱投资。
