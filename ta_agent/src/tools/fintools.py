# src/tools/financial_tools.py
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
#from pandas_ta import ta  # For technical indicators


@dataclass
class TechnicalIndicatorResult:
    indicator_name: str
    value: float
    interpretation: str

class FinancialTools:
    """
    A collection of financial analysis tools that agents can use.
    Think of this as the analyst's toolbox - containing everything from
    basic calculators to sophisticated analytical instruments.
    """
    @staticmethod
    def calculate_moving_averages(price_data: pd.Series, windows: List[int] = [20, 50, 200]) -> Dict[str, float]:
        """
        Calculates moving averages for different time windows.
        Like having multiple lenses to view price trends at different time scales.
        """
        moving_averages = {}
        for window in windows:
            ma = price_data.rolling(window=window).mean().iloc[-1]
            moving_averages[f'MA{window}'] = ma
        return moving_averages

    @staticmethod
    def calculate_rsi(price_data: pd.Series, period: int = 14) -> TechnicalIndicatorResult:
        """
        Calculates the Relative Strength Index, a momentum indicator.
        Like a thermometer for market momentum - helps identify overbought/oversold conditions.
        """
        delta = price_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # Provide interpretation
        if current_rsi > 70:
            interpretation = "Potentially overbought condition"
        elif current_rsi < 30:
            interpretation = "Potentially oversold condition"
        else:
            interpretation = "Neutral momentum"

        return TechnicalIndicatorResult(
            indicator_name="RSI",
            value=current_rsi,
            interpretation=interpretation
        )

    @staticmethod
    def analyze_volatility(price_data: pd.Series, window: int = 20) -> Dict[str, float]:
        """
        Calculates various volatility metrics.
        Like a seismograph for market movements - measures price stability.
        """
        returns = price_data.pct_change()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        return {
            'daily_volatility': returns.std(),
            'annualized_volatility': volatility,
            'rolling_volatility': returns.rolling(window=window).std().iloc[-1]
        }