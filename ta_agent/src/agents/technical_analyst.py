# src/agents/technical_analyst.py
from typing import Dict, Any
import pandas as pd
from .base_agent import BaseAgent
from ..utils.technical_indicators import TechnicalIndicators


class TechnicalAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Technical Analyst",
            description="I specialize in technical analysis of financial markets",
            model="llama3.2"
        )
        self.indicators = TechnicalIndicators()

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data and generate technical analysis.

        Args:
            input_data: Dictionary containing market data and query
        """
        # Extract price data
        prices = input_data['prices']
        close_prices = pd.Series(prices['4. close'])

        # Calculate technical indicators
        indicators = self._calculate_indicators(close_prices)

        # Generate analysis using LLM
        analysis_prompt = self._create_analysis_prompt(indicators, input_data['query'])

        response = await self.llm.generate_response(
            prompt=analysis_prompt,
            system_prompt="""You are a technical analysis expert. 
            Analyze the provided indicators and explain their implications. 
            Be specific about potential trade signals and risk levels."""
        )

        result = {
            'analysis': response,
            'indicators': indicators
        }

        self._add_to_memory({
            'role': 'analysis',
            'content': response
        })

        return result

    def _calculate_indicators(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate various technical indicators"""
        # Calculate RSI
        rsi = self.indicators.calculate_rsi(prices)

        # Calculate MACD
        macd_line, signal_line, histogram = self.indicators.calculate_macd(prices)

        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self.indicators.calculate_bollinger_bands(prices)

        return {
            'RSI': rsi.iloc[-1],  # Get latest value
            'MACD': {
                'macd_line': macd_line.iloc[-1],
                'signal_line': signal_line.iloc[-1],
                'histogram': histogram.iloc[-1]
            },
            'Bollinger_Bands': {
                'upper': upper_band.iloc[-1],
                'middle': middle_band.iloc[-1],
                'lower': lower_band.iloc[-1]
            }
        }


    def _create_analysis_prompt(
            self,
            indicators: Dict[str, Any],
            query: str
    ) -> str:
        """Create a detailed prompt for the LLM"""
        return f"""
        Based on the following technical indicators:
        
        RSI: {indicators['RSI']:.2f}
        MACD Line: {indicators['MACD']['macd_line']:.2f}
        - Signal Line: {indicators['MACD']['signal_line']:.2f}
        - Histogram: {indicators['MACD']['histogram']:.2f}
        Bollinger Bands:
        - Upper: {indicators['Bollinger_Bands']['upper']:.2f}
        - Middle: {indicators['Bollinger_Bands']['middle']:.2f}
        - Lower: {indicators['Bollinger_Bands']['lower']:.2f}
        
        Recent context:
        {self._get_memory_context()}
        
        User query: {query}
        
        Provide a detailed technical analysis considering these indicators.
        """