import pandas as pd
import numpy as np
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import sqlite3
from scipy.signal import savgol_filter
import os
from dotenv import load_dotenv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChartmatchData:
    def __init__(self, api_key: str, secret_key: str, db_path: str = 'chartmatch.db'):
        """Initialize with Alpaca credentials and database path"""
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table for raw intraday data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intraday_data (
                    symbol TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, timestamp)
                )
            ''')
            
            # Table for processed daily patterns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_patterns (
                    symbol TEXT,
                    date DATE,
                    pattern_data BLOB,    -- Serialized numpy array
                    time_points BLOB,     -- Serialized time points
                    day_open REAL,
                    day_close REAL,
                    day_high REAL,
                    day_low REAL,
                    pattern_length INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            ''')
            
            # Table to track last update
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS last_update (
                    symbol TEXT PRIMARY KEY,
                    last_date DATE
                )
            ''')
            
            conn.commit()
    
    def get_last_update(self, symbol: str) -> datetime.date:
        """Get the last update date for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT last_date FROM last_update WHERE symbol = ?', (symbol,))
            result = cursor.fetchone()
            
            if result:
                return datetime.strptime(result[0], '%Y-%m-%d').date()
            return None
    
    def update_last_update(self, symbol: str, date: datetime.date):
        """Update the last update date for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO last_update (symbol, last_date)
                VALUES (?, ?)
            ''', (symbol, date.strftime('%Y-%m-%d')))
            conn.commit()
    
    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical bar data from Alpaca"""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date
        )
        
        bars = self.data_client.get_stock_bars(request)
        
        if not hasattr(bars, 'data') or symbol not in bars.data:
            raise ValueError(f"No data returned for {symbol}")
            
        df = pd.DataFrame([{
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        } for bar in bars.data[symbol]])
        
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index).tz_convert('America/New_York')
        return df
    
    def prepare_day_pattern(self, day_data: pd.DataFrame, smooth_window: int = 21, target_points: int = 20) -> tuple:
        """Create smoothed HLC3 pattern for a single day"""
        hlc3 = (day_data['high'] + day_data['low'] + day_data['close']) / 3
        day_open = day_data['open'].iloc[0]
        pct_change = (hlc3 - day_open) / day_open * 100
        
        if len(pct_change) >= smooth_window >= 3:
            smoothed = savgol_filter(pct_change, smooth_window, 3)
        else:
            smoothed = pct_change
        
        if len(smoothed) > target_points:
            time_points = pd.date_range(
                start=day_data.index[0].replace(hour=9, minute=30),
                end=day_data.index[0].replace(hour=16, minute=0),
                periods=target_points
            ).time
            
            full_index = np.arange(len(smoothed))
            sample_index = np.linspace(0, len(smoothed)-1, target_points)
            smoothed = np.interp(sample_index, full_index, smoothed)
            
            return smoothed, time_points
        
        return smoothed, day_data.index.time
    
    def store_intraday_data(self, symbol: str, df: pd.DataFrame):
        """Store intraday data in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            # Convert DataFrame to records and ensure timestamps are strings
            df_copy = df.copy()
            df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
            records = df_copy.reset_index().to_dict('records')
            
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO intraday_data 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', [(symbol, r['timestamp'], r['open'], r['high'], r['low'], r['close'], r['volume']) 
                 for r in records])
            
            conn.commit()
    
    def store_daily_pattern(self, symbol: str, date: datetime.date, pattern: np.ndarray, 
                          times: list, day_data: pd.DataFrame):
        """Store processed daily pattern in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Serialize numpy array and time points
            pattern_bytes = pattern.tobytes()
            times_bytes = np.array([t.hour * 60 + t.minute for t in times]).tobytes()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_patterns 
                (symbol, date, pattern_data, time_points, day_open, day_close, day_high, day_low, pattern_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                date.strftime('%Y-%m-%d'),
                pattern_bytes,
                times_bytes,
                day_data['open'].iloc[0],
                day_data['close'].iloc[-1],
                day_data['high'].max(),
                day_data['low'].min(),
                len(pattern)
            ))
            
            conn.commit()
    
    def process_and_store_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None):
        """Download, process, and store data for a symbol"""
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            last_update = self.get_last_update(symbol)
            if last_update:
                start_date = datetime.combine(last_update, datetime.min.time()) + timedelta(days=1)
            else:
                # Start from January 1st, 2016
                start_date = datetime(2016, 1, 1)
        
        logger.info(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Fetch data in chunks to avoid memory issues
        chunk_size = timedelta(days=30)  # Process one month at a time
        chunk_start = start_date
        
        while chunk_start < end_date:
            chunk_end = min(chunk_start + chunk_size, end_date)
            
            try:
                df = self.fetch_historical_data(symbol, chunk_start, chunk_end)
                if df.empty:
                    chunk_start = chunk_end
                    continue
                
                # Store raw intraday data
                self.store_intraday_data(symbol, df)
                
                # Process and store daily patterns
                unique_dates = df.index.map(lambda x: x.date()).unique()
                for date in unique_dates:
                    day_data = df[df.index.date == date]
                    
                    # Filter for market hours
                    market_hours = day_data[
                        (day_data.index.time >= pd.Timestamp('09:30').time()) & 
                        (day_data.index.time <= pd.Timestamp('16:00').time())
                    ]
                    
                    if len(market_hours) >= 78:  # Minimum bars for a full day
                        pattern, times = self.prepare_day_pattern(market_hours)
                        self.store_daily_pattern(symbol, date, pattern, times, market_hours)
                
                logger.info(f"Processed chunk from {chunk_start.date()} to {chunk_end.date()}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_start.date()} to {chunk_end.date()}: {e}")
                logger.exception("Full traceback:")  # Add full traceback for better debugging
            
            chunk_start = chunk_end
        
        self.update_last_update(symbol, end_date.date())
        logger.info(f"Completed update for {symbol}")

def main():
    load_dotenv()
    
    API_KEY = os.getenv("APCA_API_KEY_ID")
    SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
    
    if not API_KEY or not SECRET_KEY:
        raise ValueError("Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your .env file")
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Initialize pattern manager with database in data directory
    manager = ChartmatchData(API_KEY, SECRET_KEY, db_path=str(data_dir / 'chartmatch.db'))
    
    # List of symbols to track
    symbols = ['SPY']  # Add more symbols as needed
    
    for symbol in symbols:
        try:
            manager.process_and_store_data(symbol)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    main() 