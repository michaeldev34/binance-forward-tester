# 🤖 Automated Trading Bot - Installation & Usage Guide

## 🎯 Complete Automated Trading System

This is a fully automated trading bot that implements a 4-component architecture:

1. **🔍 Backtester** - Tests strategies on historical data
2. **✅ Evaluator** - Validates strategies on live Binance testnet data  
3. **🎯 Combiner** - Combines strategies using Markowitz Portfolio Theory
4. **📊 Live Manager** - Manages real-time automated trading

## 📋 Prerequisites

### System Requirements
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Windows 10/11** (tested), macOS, or Linux
- **4GB RAM minimum** (8GB recommended)
- **Stable internet connection**

### Binance Testnet Account (Optional but Recommended)
1. Go to [Binance Testnet](https://testnet.binance.vision/)
2. Create a testnet account
3. Generate API keys (for enhanced features)

## 🚀 Quick Start (5 Minutes)

### Option 1: Windows One-Click Start
```bash
# 1. Double-click start_bot.bat
# 2. Follow the prompts
# 3. Bot starts automatically!
```

### Option 2: Command Line
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the bot
python run_bot.py

# 3. Follow the prompts
```

## 📦 Detailed Installation

### Step 1: Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install pandas numpy aiohttp matplotlib seaborn plotly
pip install scikit-learn jupyter asyncio-throttle python-dotenv
pip install scipy pytest
```

### Step 2: Configuration (Optional)
```bash
# Create default configuration file
python run_bot.py --create-config

# Edit bot_config.json to customize settings
```

### Step 3: Environment Setup (Optional)
Create a `.env` file for API keys:
```
BINANCE_TESTNET_API_KEY=your_testnet_api_key
BINANCE_TESTNET_SECRET=your_testnet_secret
LOG_LEVEL=INFO
```

## 🎮 Usage Examples

### Basic Usage
```bash
# Run with default settings ($100,000 capital, 4 symbols)
python run_bot.py

# Run with custom capital
python run_bot.py --capital 50000

# Run with specific symbols
python run_bot.py --symbols BTCUSDT ETHUSDT

# Run with custom configuration
python run_bot.py --config my_config.json
```

### Advanced Usage
```bash
# Dry run (simulation only)
python run_bot.py --dry-run

# Custom evaluation period
python run_bot.py --evaluation-days 7

# Custom update interval
python run_bot.py --update-interval 60

# Debug mode
python run_bot.py --log-level DEBUG
```

## ⚙️ Configuration Options

### bot_config.json Structure
```json
{
  "initial_capital": 100000.0,
  "commission_rate": 0.001,
  "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"],
  "min_sharpe_ratio": 0.8,
  "min_daily_return": 0.0045,
  "max_drawdown": 0.50,
  "evaluation_days": 14,
  "update_interval_seconds": 30,
  "emergency_stop_drawdown": 0.25
}
```

### Key Parameters
- **initial_capital**: Starting capital amount
- **symbols**: Cryptocurrency pairs to trade
- **min_sharpe_ratio**: Minimum Sharpe ratio for strategy approval (≥0.8)
- **min_daily_return**: Minimum daily return requirement (≥0.45%)
- **max_drawdown**: Maximum allowed drawdown (≤50%)
- **evaluation_days**: Days to evaluate strategies on live data
- **emergency_stop_drawdown**: Emergency stop trigger (25%)

## 🔄 Bot Workflow

### Automatic Process
1. **Strategy Creation**: Creates 5 different strategies (3 MA + 2 RSI)
2. **Backtesting**: Tests each strategy on 90 days of historical data
3. **Filtering**: Only strategies meeting criteria proceed
4. **Live Evaluation**: Tests promising strategies on live testnet data for 14 days
5. **Portfolio Optimization**: Combines approved strategies using Markowitz theory
6. **Live Trading**: Starts automated trading with real-time monitoring

### Performance Criteria
- **Sharpe Ratio**: ≥ 0.8
- **Daily Return**: ≥ 0.45%
- **Max Drawdown**: ≤ 50%
- **Consistency Score**: ≥ 0.7
- **Stability Score**: ≥ 0.6

## 📊 Monitoring & Results

### Real-time Monitoring
The bot provides live updates showing:
- Portfolio value changes
- Strategy allocations
- Daily P&L
- Performance metrics
- Risk indicators

### Results Storage
All results are saved in `bot_results/` directory:
- `backtest_results.json` - Backtesting results
- `evaluation_results.json` - Live evaluation results
- `portfolio_allocation.json` - Optimal allocation
- `final_results.json` - Complete trading session results
- `bot.log` - Detailed logs

## 🛡️ Risk Management

### Built-in Safety Features
- **Testnet Only**: Uses Binance testnet (no real money)
- **Emergency Stop**: Automatic stop at 25% drawdown
- **Position Limits**: Maximum 30% in any single strategy
- **Rebalancing**: Automatic portfolio rebalancing
- **Graceful Shutdown**: Ctrl+C for safe stopping

### Risk Controls
- Commission costs included in all calculations
- Slippage considerations
- Market data validation
- Error handling and recovery
- Comprehensive logging

## 🔧 Troubleshooting

### Common Issues

**1. "Python not found"**
```bash
# Install Python 3.8+ from python.org
# Make sure Python is in your PATH
```

**2. "Module not found"**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**3. "No data available"**
```bash
# Check internet connection
# Binance API might be temporarily unavailable
```

**4. "No strategies approved"**
```bash
# Market conditions might not favor the strategies
# Try adjusting criteria in bot_config.json
# Lower min_sharpe_ratio or min_daily_return
```

### Debug Mode
```bash
# Run with detailed logging
python run_bot.py --log-level DEBUG

# Check logs in bot_results/bot.log
```

## 📈 Performance Optimization

### Strategy Customization
Edit `bot_config.json` to add custom strategies:
```json
"strategy_configs": {
  "Custom_MA": {
    "type": "MovingAverageStrategy",
    "params": {
      "fast_period": 15,
      "slow_period": 35,
      "position_size_pct": 0.20
    }
  }
}
```

### Parameter Tuning
- Adjust evaluation periods for faster/slower validation
- Modify performance criteria based on market conditions
- Change update intervals for more/less frequent trading
- Customize position sizing for risk tolerance

## 🚨 Important Disclaimers

### Educational Purpose
- This software is for **educational and research purposes only**
- **No financial advice** is provided
- **Past performance does not guarantee future results**

### Risk Warning
- **78.18% of retail investors lose money trading CFDs**
- Cryptocurrency trading involves **substantial risk of loss**
- **Never trade with money you cannot afford to lose**
- Always test thoroughly before considering live trading

### No Warranty
- Software provided "as is" without warranty
- Authors not responsible for any financial losses
- Use at your own risk

## 🤝 Support & Community

### Getting Help
1. Check this guide first
2. Review the logs in `bot_results/bot.log`
3. Try running with `--debug` flag
4. Check GitHub issues for similar problems

### Contributing
- Fork the repository
- Create feature branches
- Add tests for new functionality
- Submit pull requests

## 📄 License

MIT License - See LICENSE file for details.

---

**🎯 Ready to start? Run `python run_bot.py` and let the automation begin!**
