#!/usr/bin/env python3
"""
Automated Trading Bot - Command Line Interface

This script provides the main entry point for running the complete automated trading bot
from the terminal. It orchestrates all 4 components automatically.

Usage:
    python run_bot.py                          # Run with default settings
    python run_bot.py --config bot_config.json # Run with custom configuration
    python run_bot.py --capital 50000          # Run with specific capital
    python run_bot.py --symbols BTCUSDT ETHUSDT # Run with specific symbols
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.automated_bot import AutomatedTradingBot, BotConfiguration


def load_configuration(config_path: str = None) -> BotConfiguration:
    """Load bot configuration from file or create default."""
    if config_path and os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return BotConfiguration(**config_dict)
    else:
        print("Using default configuration")
        return BotConfiguration()


def save_default_config(filepath: str = "bot_config.json"):
    """Save default configuration to file."""
    config = BotConfiguration()
    config_dict = {
        'initial_capital': config.initial_capital,
        'commission_rate': config.commission_rate,
        'symbols': config.symbols,
        'strategy_configs': config.strategy_configs,
        'min_sharpe_ratio': config.min_sharpe_ratio,
        'min_daily_return': config.min_daily_return,
        'max_drawdown': config.max_drawdown,
        'min_consistency_score': config.min_consistency_score,
        'min_stability_score': config.min_stability_score,
        'backtest_days': config.backtest_days,
        'evaluation_days': config.evaluation_days,
        'min_strategies_for_portfolio': config.min_strategies_for_portfolio,
        'update_interval_seconds': config.update_interval_seconds,
        'rebalance_threshold': config.rebalance_threshold,
        'rebalance_frequency': config.rebalance_frequency,
        'max_position_size': config.max_position_size,
        'emergency_stop_drawdown': config.emergency_stop_drawdown,
        'log_level': config.log_level,
        'save_results': config.save_results,
        'results_directory': config.results_directory
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Default configuration saved to {filepath}")


def print_banner():
    """Print bot startup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🤖 AUTOMATED TRADING BOT - COMPLETE SYSTEM 🤖        ║
    ║                                                              ║
    ║  4-Component Architecture:                                   ║
    ║  1. 🔍 Backtester  - Historical strategy validation         ║
    ║  2. ✅ Evaluator   - Live testnet validation               ║
    ║  3. 🎯 Combiner    - Markowitz portfolio optimization       ║
    ║  4. 📊 Live Manager - Real-time automated trading           ║
    ║                                                              ║
    ║  Performance Criteria:                                       ║
    ║  • Sharpe Ratio ≥ 0.8                                      ║
    ║  • Daily Return ≥ 0.45%                                    ║
    ║  • Max Drawdown ≤ 50%                                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_startup_info(config: BotConfiguration):
    """Print startup information."""
    print("\n🚀 BOT STARTUP INFORMATION")
    print("=" * 50)
    print(f"💰 Initial Capital: ${config.initial_capital:,.2f}")
    print(f"📊 Trading Symbols: {', '.join(config.symbols)}")
    print(f"📈 Strategies: {len(config.strategy_configs)}")
    print(f"⏱️  Update Interval: {config.update_interval_seconds}s")
    print(f"🔄 Rebalance Frequency: {config.rebalance_frequency}")
    print(f"📁 Results Directory: {config.results_directory}")
    print(f"📝 Log Level: {config.log_level}")
    print("\n🎯 PERFORMANCE CRITERIA")
    print("-" * 30)
    print(f"Sharpe Ratio: ≥ {config.min_sharpe_ratio}")
    print(f"Daily Return: ≥ {config.min_daily_return:.2%}")
    print(f"Max Drawdown: ≤ {config.max_drawdown:.2%}")
    print(f"Consistency Score: ≥ {config.min_consistency_score}")
    print(f"Stability Score: ≥ {config.min_stability_score}")
    print("\n🛡️  RISK MANAGEMENT")
    print("-" * 20)
    print(f"Max Position Size: {config.max_position_size:.2%}")
    print(f"Emergency Stop Drawdown: {config.emergency_stop_drawdown:.2%}")
    print(f"Commission Rate: {config.commission_rate:.3%}")


async def main():
    """Main entry point for the automated trading bot."""
    parser = argparse.ArgumentParser(
        description='Automated Trading Bot - Complete 4-Component System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bot.py                                    # Default settings
  python run_bot.py --config my_config.json           # Custom config
  python run_bot.py --capital 50000                   # Custom capital
  python run_bot.py --symbols BTCUSDT ETHUSDT ADAUSDT # Custom symbols
  python run_bot.py --create-config                   # Create default config file
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--capital', 
        type=float, 
        help='Initial capital amount (overrides config)'
    )
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        help='Trading symbols (overrides config)'
    )
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        help='Logging level (overrides config)'
    )
    parser.add_argument(
        '--results-dir', 
        type=str, 
        help='Results directory (overrides config)'
    )
    parser.add_argument(
        '--create-config', 
        action='store_true', 
        help='Create default configuration file and exit'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Run in simulation mode (no real trades)'
    )
    parser.add_argument(
        '--evaluation-days', 
        type=int, 
        help='Number of days for strategy evaluation'
    )
    parser.add_argument(
        '--update-interval', 
        type=int, 
        help='Update interval in seconds'
    )
    
    args = parser.parse_args()
    
    # Handle create-config option
    if args.create_config:
        save_default_config()
        return
    
    # Print banner
    print_banner()
    
    # Load configuration
    config = load_configuration(args.config)
    
    # Override config with command line arguments
    if args.capital:
        config.initial_capital = args.capital
    if args.symbols:
        config.symbols = args.symbols
    if args.log_level:
        config.log_level = args.log_level
    if args.results_dir:
        config.results_directory = args.results_dir
    if args.evaluation_days:
        config.evaluation_days = args.evaluation_days
    if args.update_interval:
        config.update_interval_seconds = args.update_interval
    
    # Print startup information
    print_startup_info(config)
    
    # Create results directory
    os.makedirs(config.results_directory, exist_ok=True)
    
    # Confirmation prompt
    print(f"\n⚠️  WARNING: This bot will trade with ${config.initial_capital:,.2f} on Binance testnet")
    print("The bot will run autonomously and make trading decisions automatically.")
    
    if not args.dry_run:
        response = input("\nDo you want to continue? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("Bot startup cancelled by user.")
            return
    else:
        print("\n🔍 Running in DRY-RUN mode (simulation only)")
    
    # Create and run the bot
    print(f"\n🤖 Initializing Automated Trading Bot...")
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        bot = AutomatedTradingBot(config)
        
        print("\n🚀 Starting complete automation pipeline...")
        print("Press Ctrl+C to stop the bot gracefully")
        print("=" * 60)
        
        await bot.run_complete_automation()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Bot stopped by user (Ctrl+C)")
        print("Graceful shutdown completed.")
        
    except Exception as e:
        print(f"\n\n💥 Bot crashed with error: {e}")
        print("Check the logs for more details.")
        sys.exit(1)
    
    finally:
        print(f"\n📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🏁 Automated Trading Bot session completed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot interrupted during startup.")
    except Exception as e:
        print(f"Failed to start bot: {e}")
        sys.exit(1)
