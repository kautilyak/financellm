# src/data_providers/alpha_vantage.py
import asyncio
from typing import Dict, Any
import time
from alpha_vantage.timeseries import TimeSeries
import pandas as pd


class AlphaVantageProvider:
    def __init__(self, api_key: str):
        self.client = TimeSeries(key=api_key, output_format='pandas')
        self.last_call_timestamp = 0
        self.min_call_interval = 12  # Alpha Vantage free tier: 5 calls per minute

    async def _throttle(self):
        """Implement rate limiting to avoid API throttling"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_timestamp

        if time_since_last_call < self.min_call_interval:
            await asyncio.sleep(self.min_call_interval - time_since_last_call)

        self.last_call_timestamp = time.time()

    async def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get daily stock data with error handling"""
        await self._throttle()

        try:
            data, meta_data = self.client.get_daily(symbol=symbol, outputsize='compact')

            # Convert data to a more manageable format
            processed_data = {
                'prices': data,
                'metadata': meta_data,
                'summary': self._generate_summary(data)

            }

            return processed_data

        except Exception as e:
            raise ValueError(f"Error fetching data for {symbol}: {str(e)}")


    def _generate_summary(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate summary statistics for the stock data"""
        return {
            'latest_close': data['4. close'].iloc[0],
            'avg_volume': data['5. volume'].mean(),
            'price_change_percent': ((data['4. close'].iloc[0] - data['4. close'].iloc[-1])
                                     / data['4. close'].iloc[-1] * 100)
        }



