# src/utils/technical_indicators.py
import pandas as pd
import numpy as np


class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            data: Series of prices
            periods: RSI period (default: 14)
        """
        # Calculate price changes
        delta = data.diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(data: pd.Series,
                       fast_period: int = 12,
                       slow_period: int = 26,
                       signal_period: int = 9) -> tuple:
        """
        Calculate Moving Average Convergence Divergence (MACD).

        Args:
            data: Series of prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        """
        # Calculate EMAs
        fast_ema = data.ewm(span=fast_period, adjust=False).mean()
        slow_ema = data.ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series,
                                  periods: int = 20,
                                  num_std: float = 2) -> tuple:
        """
        Calculate Bollinger Bands.

        Args:
            data: Series of prices
            periods: Moving average period
            num_std: Number of standard deviations
        """
        # Calculate middle band (SMA)
        middle_band = data.rolling(window=periods).mean()

        # Calculate standard deviation
        std = data.rolling(window=periods).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        return upper_band, middle_band, lower_band