"""
Final Component: Automated Trading Bot - Complete System Orchestrator

This module provides the final automation layer that orchestrates all 4 components
into a fully automated trading bot that can run continuously from the terminal.
"""

import pandas as pd
import numpy as np
import asyncio
import logging
import json
import os
import signal
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import argparse

from .backtester import SingleStrategyBacktester, BacktestMetrics
from .evaluator import StrategyEvaluator, EvaluationResult
from .combiner import StrategyCombiner, PortfolioAllocation
from .live_manager import LiveAllocationManager, LiveAllocation
from .base_strategy import BaseStrategy
from ..strategies.moving_average import MovingAverageStrategy
from ..strategies.rsi_strategy import RSIStrategy
from ..data.data_manager import DataManager


@dataclass
class BotConfiguration:
    """Configuration for the automated trading bot."""
    # Capital settings
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    
    # Strategy settings
    symbols: List[str] = None
    strategy_configs: Dict = None
    
    # Performance criteria
    min_sharpe_ratio: float = 0.8
    min_daily_return: float = 0.0045  # 0.45%
    max_drawdown: float = 0.50
    min_consistency_score: float = 0.7
    min_stability_score: float = 0.6
    
    # Evaluation settings
    backtest_days: int = 90
    evaluation_days: int = 14
    min_strategies_for_portfolio: int = 2
    
    # Live trading settings
    update_interval_seconds: int = 30
    rebalance_threshold: float = 0.05
    rebalance_frequency: str = 'weekly'
    
    # Risk management
    max_position_size: float = 0.3  # Max 30% in any single strategy
    emergency_stop_drawdown: float = 0.25  # Stop if 25% drawdown
    
    # Logging and monitoring
    log_level: str = 'INFO'
    save_results: bool = True
    results_directory: str = 'bot_results'
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
        
        if self.strategy_configs is None:
            self.strategy_configs = {
                'MA_Fast': {
                    'type': 'MovingAverageStrategy',
                    'params': {'fast_period': 10, 'slow_period': 30, 'position_size_pct': 0.15}
                },
                'MA_Medium': {
                    'type': 'MovingAverageStrategy', 
                    'params': {'fast_period': 20, 'slow_period': 50, 'position_size_pct': 0.12}
                },
                'MA_Slow': {
                    'type': 'MovingAverageStrategy',
                    'params': {'fast_period': 50, 'slow_period': 100, 'position_size_pct': 0.10}
                },
                'RSI_Conservative': {
                    'type': 'RSIStrategy',
                    'params': {'rsi_period': 14, 'oversold_level': 25, 'overbought_level': 75, 'position_size_pct': 0.08}
                },
                'RSI_Aggressive': {
                    'type': 'RSIStrategy',
                    'params': {'rsi_period': 14, 'oversold_level': 35, 'overbought_level': 65, 'position_size_pct': 0.12}
                }
            }


class AutomatedTradingBot:
    """
    Complete automated trading bot that orchestrates all 4 components.
    
    This is the final class that provides full automation:
    1. Automatically creates and backtests strategies
    2. Evaluates them on live testnet data
    3. Combines approved strategies using Markowitz optimization
    4. Runs live portfolio management continuously
    """
    
    def __init__(self, config: BotConfiguration):
        """
        Initialize the automated trading bot.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        self.is_running = False
        self.emergency_stop = False
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.backtester = SingleStrategyBacktester(
            initial_capital=config.initial_capital,
            commission=config.commission_rate
        )
        
        self.evaluator = StrategyEvaluator(
            initial_capital=config.initial_capital * 0.5,  # Use half capital for evaluation
            commission=config.commission_rate,
            min_evaluation_days=config.evaluation_days,
            max_evaluation_days=config.evaluation_days * 2
        )
        
        self.combiner = StrategyCombiner(
            target_sharpe=config.min_sharpe_ratio,
            target_daily_return=config.min_daily_return,
            max_drawdown_limit=config.max_drawdown,
            rebalance_frequency=config.rebalance_frequency
        )
        
        self.live_manager = LiveAllocationManager(
            initial_capital=config.initial_capital,
            commission=config.commission_rate,
            update_interval_seconds=config.update_interval_seconds,
            rebalance_threshold=config.rebalance_threshold
        )
        
        # Data manager
        self.data_manager = DataManager()
        
        # Bot state
        self.strategies: Dict[str, BaseStrategy] = {}
        self.backtest_results: Dict[str, BacktestMetrics] = {}
        self.evaluation_results: Dict[str, EvaluationResult] = {}
        self.approved_strategies: List[EvaluationResult] = []
        self.current_allocation: Optional[PortfolioAllocation] = None
        
        # Results storage
        self.results_dir = config.results_directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.results_directory}/bot.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
    
    async def run_complete_automation(self):
        """
        Run the complete automated trading pipeline.
        
        This is the main method that orchestrates all 4 components automatically.
        """
        self.logger.info("🚀 Starting Complete Automated Trading Bot")
        self.logger.info("=" * 60)
        
        try:
            self.is_running = True
            
            # Phase 1: Strategy Creation and Backtesting
            self.logger.info("📊 Phase 1: Creating and Backtesting Strategies")
            await self._phase1_backtest_strategies()
            
            # Phase 2: Live Evaluation on Testnet
            self.logger.info("🔍 Phase 2: Evaluating Strategies on Live Testnet Data")
            await self._phase2_evaluate_strategies()
            
            # Phase 3: Portfolio Optimization
            self.logger.info("🎯 Phase 3: Creating Optimal Portfolio")
            await self._phase3_optimize_portfolio()
            
            # Phase 4: Live Trading Automation
            self.logger.info("📈 Phase 4: Starting Live Automated Trading")
            await self._phase4_live_trading()
            
        except Exception as e:
            self.logger.error(f"Critical error in automation pipeline: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _phase1_backtest_strategies(self):
        """Phase 1: Create and backtest all strategies."""
        self.logger.info("Creating strategies from configuration...")
        
        # Create strategies
        self.strategies = self._create_strategies_from_config()
        self.logger.info(f"Created {len(self.strategies)} strategies")
        
        # Get historical data
        self.logger.info("Fetching historical data for backtesting...")
        await self.data_manager.initialize()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.backtest_days)
        
        historical_data = await self._fetch_historical_data(start_date, end_date)
        
        # Backtest each strategy
        self.logger.info("Starting backtesting phase...")
        for name, strategy in self.strategies.items():
            self.logger.info(f"Backtesting {name}...")
            
            try:
                metrics = self.backtester.backtest_strategy(strategy, historical_data)
                self.backtest_results[name] = metrics
                
                self.logger.info(f"{name} - Return: {metrics.total_return:.2%}, "
                               f"Sharpe: {metrics.sharpe_ratio:.2f}, "
                               f"Drawdown: {metrics.max_drawdown:.2%}")
                
                if metrics.meets_criteria(
                    min_sharpe=self.config.min_sharpe_ratio,
                    min_daily_return=self.config.min_daily_return,
                    max_drawdown=self.config.max_drawdown
                ):
                    self.logger.info(f"✅ {name} passed backtesting criteria")
                else:
                    self.logger.info(f"❌ {name} failed backtesting criteria")
                    
            except Exception as e:
                self.logger.error(f"Error backtesting {name}: {e}")
        
        # Save backtest results
        if self.config.save_results:
            self._save_backtest_results()
    
    async def _phase2_evaluate_strategies(self):
        """Phase 2: Evaluate promising strategies on live testnet data."""
        # Filter strategies that passed backtesting
        promising_strategies = {
            name: strategy for name, strategy in self.strategies.items()
            if name in self.backtest_results and 
            self.backtest_results[name].meets_criteria(
                min_sharpe=self.config.min_sharpe_ratio,
                min_daily_return=self.config.min_daily_return,
                max_drawdown=self.config.max_drawdown
            )
        }
        
        self.logger.info(f"Evaluating {len(promising_strategies)} promising strategies on live data")
        
        # Set evaluation criteria
        self.evaluator.criteria.update({
            'min_sharpe_ratio': self.config.min_sharpe_ratio,
            'min_daily_return': self.config.min_daily_return,
            'max_drawdown': self.config.max_drawdown,
            'min_consistency_score': self.config.min_consistency_score,
            'min_stability_score': self.config.min_stability_score
        })
        
        # Evaluate each promising strategy
        for name, strategy in promising_strategies.items():
            self.logger.info(f"Live evaluating {name}...")
            
            try:
                evaluation_result = await self.evaluator.evaluate_strategy(
                    strategy=strategy,
                    historical_metrics=self.backtest_results[name],
                    evaluation_days=self.config.evaluation_days
                )
                
                self.evaluation_results[name] = evaluation_result
                
                if evaluation_result.is_approved:
                    self.approved_strategies.append(evaluation_result)
                    self.logger.info(f"✅ {name} approved for portfolio inclusion")
                else:
                    self.logger.warning(f"❌ {name} rejected: {', '.join(evaluation_result.rejection_reasons)}")
                    
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {e}")
        
        self.logger.info(f"Final approved strategies: {len(self.approved_strategies)}")
        
        # Save evaluation results
        if self.config.save_results:
            self._save_evaluation_results()
    
    async def _phase3_optimize_portfolio(self):
        """Phase 3: Create optimal portfolio from approved strategies."""
        if len(self.approved_strategies) < self.config.min_strategies_for_portfolio:
            raise ValueError(f"Insufficient approved strategies ({len(self.approved_strategies)}) "
                           f"for portfolio optimization (minimum: {self.config.min_strategies_for_portfolio})")
        
        self.logger.info(f"Creating optimal portfolio from {len(self.approved_strategies)} approved strategies")
        
        try:
            self.current_allocation = self.combiner.create_optimal_portfolio(
                approved_strategies=self.approved_strategies,
                lookback_days=30
            )
            
            self.logger.info("Portfolio optimization completed:")
            self.logger.info(f"Expected Return: {self.current_allocation.expected_return:.2%}")
            self.logger.info(f"Expected Volatility: {self.current_allocation.expected_volatility:.2%}")
            self.logger.info(f"Sharpe Ratio: {self.current_allocation.sharpe_ratio:.2f}")
            self.logger.info(f"Strategy Weights: {self.current_allocation.strategy_weights}")
            
            # Validate allocation
            is_valid, issues = self.combiner.validate_allocation(self.current_allocation)
            if not is_valid:
                self.logger.warning(f"Portfolio allocation has issues: {issues}")
            
            # Save allocation
            if self.config.save_results:
                self._save_portfolio_allocation()
                
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            raise
    
    async def _phase4_live_trading(self):
        """Phase 4: Start live automated trading."""
        if not self.current_allocation:
            raise ValueError("No portfolio allocation available for live trading")
        
        # Prepare strategies for live trading
        live_strategies = {}
        for result in self.approved_strategies:
            strategy_name = result.strategy_name
            live_strategies[strategy_name] = self.strategies[strategy_name]
        
        # Initialize live manager
        await self.live_manager.initialize(
            strategies=live_strategies,
            initial_allocation=self.current_allocation
        )
        
        # Add monitoring callbacks
        self.live_manager.add_update_callback(self._live_monitoring_callback)
        self.live_manager.add_update_callback(self._emergency_stop_callback)
        
        self.logger.info("🔴 STARTING LIVE AUTOMATED TRADING")
        self.logger.info("Bot is now running autonomously...")
        self.logger.info("Press Ctrl+C to stop gracefully")
        
        # Start live management
        await self.live_manager.start_live_management()
    
    def _create_strategies_from_config(self) -> Dict[str, BaseStrategy]:
        """Create strategy instances from configuration."""
        strategies = {}
        
        for name, config in self.config.strategy_configs.items():
            strategy_type = config['type']
            params = config['params']
            
            if strategy_type == 'MovingAverageStrategy':
                strategy = MovingAverageStrategy(
                    name=name,
                    symbols=self.config.symbols,
                    **params
                )
            elif strategy_type == 'RSIStrategy':
                strategy = RSIStrategy(
                    name=name,
                    symbols=self.config.symbols,
                    **params
                )
            else:
                self.logger.warning(f"Unknown strategy type: {strategy_type}")
                continue
            
            strategies[name] = strategy
        
        return strategies
    
    async def _fetch_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical data for all symbols."""
        all_data = []
        
        for symbol in self.config.symbols:
            try:
                data = await self.data_manager.get_historical_data(
                    symbol=symbol,
                    interval='1h',
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not data.empty:
                    data['symbol'] = symbol
                    all_data.append(data)
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
        
        if not all_data:
            raise ValueError("No historical data available")
        
        return pd.concat(all_data, ignore_index=True)
    
    async def _live_monitoring_callback(self, live_allocation: LiveAllocation):
        """Callback for live monitoring and logging."""
        self.logger.info(f"💰 Portfolio Value: ${live_allocation.total_portfolio_value:,.2f} | "
                        f"Daily P&L: ${live_allocation.daily_pnl:,.2f} | "
                        f"Unrealized P&L: ${live_allocation.unrealized_pnl:,.2f}")
        
        # Log strategy allocations
        for strategy, allocation in live_allocation.strategy_allocations.items():
            self.logger.debug(f"  {strategy}: {allocation:.2%}")
    
    async def _emergency_stop_callback(self, live_allocation: LiveAllocation):
        """Emergency stop callback for risk management."""
        if live_allocation.performance_metrics:
            current_drawdown = live_allocation.performance_metrics.get('max_drawdown', 0)
            
            if abs(current_drawdown) > self.config.emergency_stop_drawdown:
                self.logger.critical(f"🚨 EMERGENCY STOP TRIGGERED! "
                                   f"Drawdown {current_drawdown:.2%} exceeds limit {self.config.emergency_stop_drawdown:.2%}")
                self.emergency_stop = True
                self.stop()
    
    def _save_backtest_results(self):
        """Save backtest results to file."""
        results_data = {}
        for name, metrics in self.backtest_results.items():
            results_data[name] = asdict(metrics)
        
        filepath = os.path.join(self.results_dir, 'backtest_results.json')
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Backtest results saved to {filepath}")
    
    def _save_evaluation_results(self):
        """Save evaluation results to file."""
        results_data = {}
        for name, result in self.evaluation_results.items():
            results_data[name] = result.to_dict()
        
        filepath = os.path.join(self.results_dir, 'evaluation_results.json')
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation results saved to {filepath}")
    
    def _save_portfolio_allocation(self):
        """Save portfolio allocation to file."""
        if self.current_allocation:
            filepath = os.path.join(self.results_dir, 'portfolio_allocation.json')
            with open(filepath, 'w') as f:
                json.dump(self.current_allocation.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Portfolio allocation saved to {filepath}")
    
    def stop(self):
        """Stop the automated trading bot."""
        self.logger.info("Stopping automated trading bot...")
        self.is_running = False
        
        if hasattr(self, 'live_manager'):
            self.live_manager.stop()
    
    async def _cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up resources...")
        
        # Save final results
        if self.config.save_results and hasattr(self, 'live_manager'):
            try:
                final_results_path = os.path.join(self.results_dir, 'final_results.json')
                self.live_manager.export_live_data(final_results_path)
                self.logger.info(f"Final results saved to {final_results_path}")
            except Exception as e:
                self.logger.error(f"Error saving final results: {e}")
        
        self.logger.info("🏁 Automated Trading Bot Shutdown Complete")


def main():
    """Main entry point for the automated trading bot."""
    parser = argparse.ArgumentParser(description='Automated Trading Bot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], help='Trading symbols')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = BotConfiguration(**config_dict)
    else:
        config = BotConfiguration(
            initial_capital=args.capital,
            symbols=args.symbols,
            log_level=args.log_level
        )
    
    # Create and run bot
    bot = AutomatedTradingBot(config)
    
    try:
        asyncio.run(bot.run_complete_automation())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Bot crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
