"""
RSI Strategy Implementation

A Relative Strength Index (RSI) based trading strategy using pandas DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from ..core.base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    RSI-based trading strategy.
    
    Generates buy signals when RSI is oversold and sell signals when overbought.
    Uses pandas DataFrames for efficient calculation and signal generation.
    """
    
    def __init__(self, 
                 name: str = "RSI_Strategy",
                 symbols: List[str] = None,
                 timeframe: str = '1h',
                 rsi_period: int = 14,
                 oversold_level: float = 30,
                 overbought_level: float = 70,
                 position_size_pct: float = 0.1):
        """
        Initialize RSI Strategy.
        
        Args:
            name: Strategy name
            symbols: List of trading symbols
            timeframe: Trading timeframe
            rsi_period: RSI calculation period
            oversold_level: RSI level considered oversold (buy signal)
            overbought_level: RSI level considered overbought (sell signal)
            position_size_pct: Position size as percentage of portfolio
        """
        super().__init__(name, symbols or [], timeframe)
        
        self.rsi_period = rsi_period
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.position_size_pct = position_size_pct
        
        # Store parameters
        self.parameters = {
            'rsi_period': rsi_period,
            'oversold_level': oversold_level,
            'overbought_level': overbought_level,
            'position_size_pct': position_size_pct
        }
        
        # Strategy state
        self.last_signals = {}
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add RSI and related indicators to the data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with RSI indicators
        """
        df = data.copy()
        
        if 'close' not in df.columns or len(df) < self.rsi_period + 1:
            return df
        
        try:
            # Calculate RSI
            df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
            
            # RSI levels
            df['rsi_oversold'] = self.oversold_level
            df['rsi_overbought'] = self.overbought_level
            
            # RSI signals
            df['rsi_buy_signal'] = df['rsi'] < self.oversold_level
            df['rsi_sell_signal'] = df['rsi'] > self.overbought_level
            
            # RSI momentum
            df['rsi_momentum'] = df['rsi'].diff()
            
            # RSI divergence (simplified)
            df['price_momentum'] = df['close'].pct_change()
            df['rsi_divergence'] = np.where(
                (df['price_momentum'] > 0) & (df['rsi_momentum'] < 0), -1,  # Bearish divergence
                np.where((df['price_momentum'] < 0) & (df['rsi_momentum'] > 0), 1, 0)  # Bullish divergence
            )
            
            # RSI trend
            df['rsi_trend'] = np.where(df['rsi'] > 50, 1, -1)
            
            # Volume confirmation if available
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
            
        except Exception as e:
            print(f"Error adding RSI indicators: {e}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI using pandas for efficient computation.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI levels.
        
        Args:
            data: OHLCV data with indicators
            
        Returns:
            DataFrame with signals
        """
        signals = []
        
        if data.empty or 'close' not in data.columns:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'signal', 'price', 'confidence'])
        
        # Add indicators if not present
        if 'rsi' not in data.columns:
            data = self.add_indicators(data)
        
        # Group by symbol to process each separately
        for symbol in data['symbol'].unique() if 'symbol' in data.columns else [None]:
            symbol_data = data[data['symbol'] == symbol] if symbol else data
            symbol_data = symbol_data.sort_values('timestamp' if 'timestamp' in symbol_data.columns else symbol_data.index)
            
            # Skip if insufficient data
            if len(symbol_data) < self.rsi_period + 5:
                continue
            
            # Get the latest few rows to check for signals
            recent_data = symbol_data.tail(10)  # Look at last 10 periods
            
            for idx, row in recent_data.iterrows():
                signal_value = 0
                confidence = 0.0
                
                # Check for valid RSI value
                rsi_value = row.get('rsi')
                if pd.isna(rsi_value):
                    continue
                
                # Check for oversold condition (buy signal)
                if rsi_value < self.oversold_level:
                    # Additional confirmation checks
                    rsi_momentum = row.get('rsi_momentum', 0)
                    volume_ratio = row.get('volume_ratio', 1)
                    
                    # Stronger signal if RSI is turning up from oversold
                    if rsi_momentum > 0:
                        signal_value = 1
                        confidence = min(0.9, (self.oversold_level - rsi_value) / self.oversold_level + 0.3)
                    elif rsi_value < self.oversold_level * 0.8:  # Very oversold
                        signal_value = 1
                        confidence = min(0.8, (self.oversold_level - rsi_value) / self.oversold_level + 0.2)
                    
                    # Volume confirmation
                    if volume_ratio > 1.2:
                        confidence = min(1.0, confidence * 1.2)
                
                # Check for overbought condition (sell signal)
                elif rsi_value > self.overbought_level:
                    rsi_momentum = row.get('rsi_momentum', 0)
                    volume_ratio = row.get('volume_ratio', 1)
                    
                    # Stronger signal if RSI is turning down from overbought
                    if rsi_momentum < 0:
                        signal_value = -1
                        confidence = min(0.9, (rsi_value - self.overbought_level) / (100 - self.overbought_level) + 0.3)
                    elif rsi_value > self.overbought_level * 1.2:  # Very overbought
                        signal_value = -1
                        confidence = min(0.8, (rsi_value - self.overbought_level) / (100 - self.overbought_level) + 0.2)
                    
                    # Volume confirmation
                    if volume_ratio > 1.2:
                        confidence = min(1.0, confidence * 1.2)
                
                # Check for divergence signals
                divergence = row.get('rsi_divergence', 0)
                if divergence != 0 and abs(signal_value) == 0:
                    signal_value = divergence
                    confidence = 0.4  # Lower confidence for divergence signals
                
                # Only generate signal if confidence is above threshold
                if abs(signal_value) > 0 and confidence > 0.3:
                    # Check if this is a new signal (avoid duplicate signals)
                    last_signal = self.last_signals.get(symbol, 0)
                    if signal_value != last_signal:
                        signals.append({
                            'timestamp': row.get('timestamp', datetime.now()),
                            'symbol': symbol or 'UNKNOWN',
                            'signal': signal_value,
                            'price': row['close'],
                            'confidence': confidence
                        })
                        
                        # Update last signal
                        self.last_signals[symbol] = signal_value
        
        return pd.DataFrame(signals)
    
    def calculate_position_size(self, signal: pd.Series, portfolio_value: float) -> float:
        """
        Calculate position size based on RSI signal strength.
        
        Args:
            signal: Signal row from signals DataFrame
            portfolio_value: Current portfolio value
            
        Returns:
            Position size in base currency
        """
        try:
            # Base position size
            base_position_value = portfolio_value * self.position_size_pct
            
            # Adjust based on confidence
            confidence = signal.get('confidence', 0.5)
            adjusted_position_value = base_position_value * confidence
            
            # Additional adjustment based on signal strength
            # For RSI, stronger oversold/overbought conditions get larger positions
            if confidence > 0.7:
                adjusted_position_value *= 1.2  # Increase by 20% for high confidence
            elif confidence < 0.4:
                adjusted_position_value *= 0.8  # Decrease by 20% for low confidence
            
            # Convert to quantity
            price = signal['price']
            quantity = adjusted_position_value / price
            
            # Apply signal direction
            signal_value = signal['signal']
            return quantity * signal_value
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.0
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information and parameters."""
        return {
            'name': self.name,
            'type': 'RSI Strategy',
            'description': f'RSI({self.rsi_period}) with levels {self.oversold_level}/{self.overbought_level}',
            'parameters': self.parameters,
            'symbols': self.symbols,
            'timeframe': self.timeframe
        }
    
    def optimize_parameters(self, historical_data: pd.DataFrame, 
                          rsi_period_range: tuple = (10, 20), 
                          oversold_range: tuple = (20, 35),
                          overbought_range: tuple = (65, 80)) -> Dict:
        """
        Simple parameter optimization using historical data.
        
        Args:
            historical_data: Historical OHLCV data
            rsi_period_range: Range for RSI period (min, max)
            oversold_range: Range for oversold level (min, max)
            overbought_range: Range for overbought level (min, max)
            
        Returns:
            Dictionary with optimal parameters and performance
        """
        best_params = None
        best_performance = -np.inf
        results = []
        
        for rsi_period in range(rsi_period_range[0], rsi_period_range[1] + 1, 2):
            for oversold in range(oversold_range[0], oversold_range[1] + 1, 5):
                for overbought in range(overbought_range[0], overbought_range[1] + 1, 5):
                    if oversold >= overbought:
                        continue
                    
                    # Create temporary strategy with these parameters
                    temp_strategy = RSIStrategy(
                        name=f"RSI_{rsi_period}_{oversold}_{overbought}",
                        symbols=self.symbols,
                        rsi_period=rsi_period,
                        oversold_level=oversold,
                        overbought_level=overbought,
                        position_size_pct=self.position_size_pct
                    )
                    
                    # Test on historical data
                    data_with_indicators = temp_strategy.add_indicators(historical_data)
                    signals = temp_strategy.generate_signals(data_with_indicators)
                    
                    # Calculate simple performance metric
                    if not signals.empty:
                        # Simple return calculation
                        returns = []
                        for _, signal in signals.iterrows():
                            if signal['signal'] != 0:
                                # Simulate holding for 24 periods (hours)
                                entry_price = signal['price']
                                # Find price 24 periods later (simplified)
                                future_idx = min(len(historical_data) - 1, 
                                               historical_data.index.get_loc(historical_data.index[-1]) + 24)
                                exit_price = historical_data.iloc[future_idx]['close']
                                
                                if signal['signal'] > 0:  # Long position
                                    ret = (exit_price - entry_price) / entry_price
                                else:  # Short position
                                    ret = (entry_price - exit_price) / entry_price
                                
                                returns.append(ret)
                        
                        if returns:
                            avg_return = np.mean(returns)
                            win_rate = len([r for r in returns if r > 0]) / len(returns)
                            performance = avg_return * win_rate  # Simple performance metric
                            
                            results.append({
                                'rsi_period': rsi_period,
                                'oversold_level': oversold,
                                'overbought_level': overbought,
                                'avg_return': avg_return,
                                'win_rate': win_rate,
                                'performance': performance,
                                'num_signals': len(signals)
                            })
                            
                            if performance > best_performance:
                                best_performance = performance
                                best_params = {
                                    'rsi_period': rsi_period,
                                    'oversold_level': oversold,
                                    'overbought_level': overbought
                                }
        
        return {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'all_results': pd.DataFrame(results)
        }
