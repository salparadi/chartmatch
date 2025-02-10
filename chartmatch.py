import pandas as pd
import numpy as np
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
import os
from dotenv import load_dotenv
import logging
import random
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration Constants
TARGET_DATE = '2025-01-17'   # Date to analyze
TARGET_START_TIME = '09:30'  # Start of analysis window
TARGET_END_TIME = '10:30'    # End of analysis window
TARGET_SYMBOL = 'SPY'        # Symbol to analyze
MATCH_COUNT = 10             # Number of similar patterns to find
SMOOTH_WINDOW = 21           # Window size for pattern smoothing
TARGET_POINTS = 20           # Number of points to reduce pattern to
SIMILARITY_THRESHOLD = 0.8   # Minimum similarity score to consider a match

# Data Storage Constants
DATA_DIR = Path('data')                    # Base directory for all data
DB_PATH = DATA_DIR / 'chartmatch.db'       # SQLite database path

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)

class Chartmatch:
    def __init__(self, api_key: str, secret_key: str):
        """Initialize with Alpaca credentials"""
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
    
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
    
    def prepare_day_pattern(self, day_data: pd.DataFrame, 
                          smooth_window: int = SMOOTH_WINDOW, 
                          target_points: int = TARGET_POINTS) -> tuple:
        """Create smoothed HLC3 pattern for a single day with reduced points"""
        # Calculate HLC3
        hlc3 = (day_data['high'] + day_data['low'] + day_data['close']) / 3
        
        # Convert to percentage change from day's open
        day_open = day_data['open'].iloc[0]
        pct_change = (hlc3 - day_open) / day_open * 100
        
        # First apply smoothing with larger window
        if len(pct_change) >= smooth_window >= 3:
            smoothed = savgol_filter(pct_change, smooth_window, 3)
        else:
            smoothed = pct_change
        
        # Then downsample to target number of points
        if len(smoothed) > target_points:
            # Create time points for market hours
            time_points = pd.date_range(
                start=day_data.index[0].replace(hour=9, minute=30),
                end=day_data.index[0].replace(hour=16, minute=0),
                periods=target_points
            ).time
            
            # Use linear interpolation to reduce points
            full_index = np.arange(len(smoothed))
            sample_index = np.linspace(0, len(smoothed)-1, target_points)
            smoothed = np.interp(sample_index, full_index, smoothed)
            
            return smoothed, time_points
        
        return smoothed, day_data.index.time
    
    def get_random_day_pattern(self, df: pd.DataFrame) -> tuple:
        """Get a random day's pattern from the dataset"""
        market_open = pd.Timestamp('09:30').time()
        market_close = pd.Timestamp('16:00').time()
        
        # Get unique dates using pandas
        unique_dates = df.index.map(lambda x: x.date()).unique()
        
        while True:
            random_date = random.choice(unique_dates)
            # Get data for this date
            day_data = df[df.index.date == random_date]
            
            # Filter for market hours
            market_hours = day_data[
                (day_data.index.time >= market_open) & 
                (day_data.index.time <= market_close)
            ]
            
            if len(market_hours) >= 78:  # Minimum bars for a full day
                pattern, times = self.prepare_day_pattern(market_hours)
                return random_date, pattern, market_hours, times
            
            # Remove this date from consideration if it doesn't have enough data
            unique_dates = unique_dates[unique_dates != random_date]
            if len(unique_dates) == 0:
                raise ValueError("No valid trading days found with sufficient data")
    
    def plot_patterns(self, patterns: list):
        """Plot multiple patterns using Plotly"""
        fig = go.Figure()
        
        # Use a better color palette for more patterns
        colors = [
            '#1f77b4',  # blue
            '#d62728',  # red
            '#2ca02c',  # green
            '#9467bd',  # purple
            '#ff7f0e',  # orange
            '#17becf',  # cyan
            '#e377c2',  # pink
            '#bcbd22',  # olive
            '#7f7f7f',  # gray
            '#8c564b',  # brown
        ]
        
        # Plot target pattern first (it's always the first one in the list)
        date, pattern, raw_data, times = patterns[0]
        time_labels = [t.strftime('%H:%M') for t in times]
        
        # Get the end time index for splitting the pattern
        end_time = pd.Timestamp(TARGET_END_TIME).time()
        split_idx = next((i for i, t in enumerate(times) if t > end_time), len(times))
        
        # Plot target pattern up to end_time (dotted)
        fig.add_trace(go.Scatter(
            x=time_labels[:split_idx],
            y=pattern[:split_idx],
            name=f'TARGET: {date} (Analysis Window)',
            line=dict(
                color='black',
                width=3,
                dash='dot',
                shape='spline',
                smoothing=1.3
            ),
            mode='lines',
            hovertemplate=f'Time: %{{x}}<br>Change: %{{y:.2f}}%<extra>TARGET: {date}</extra>'
        ))
        
        # Plot remainder of target pattern (dashed)
        if split_idx < len(pattern):
            fig.add_trace(go.Scatter(
                x=time_labels[split_idx-1:],  # Overlap one point for continuity
                y=pattern[split_idx-1:],      # Overlap one point for continuity
                name=f'TARGET: {date} (Remainder)',
                line=dict(
                    color='black',
                    width=2,
                    dash='dash',
                    shape='spline',
                    smoothing=1.3
                ),
                mode='lines',
                hovertemplate=f'Time: %{{x}}<br>Change: %{{y:.2f}}%<extra>TARGET: {date}</extra>'
            ))
        
        # Add candlestick for target if needed
        candle_times = pd.date_range(
            start=pd.Timestamp('09:30'),
            end=pd.Timestamp('16:00'),
            periods=len(pattern)
        ).time
        
        candle_df = pd.DataFrame({
            'time': [t.strftime('%H:%M') for t in candle_times],
            'open': raw_data['open'],
            'high': raw_data['high'],
            'low': raw_data['low'],
            'close': raw_data['close']
        })
        
        fig.add_trace(go.Candlestick(
            x=candle_df['time'],
            open=candle_df['open'],
            high=candle_df['high'],
            low=candle_df['low'],
            close=candle_df['close'],
            name=f'Candles {date}',
            opacity=0.15,
            visible='legendonly',
            showlegend=True
        ))
        
        # Plot similar patterns
        for (date, pattern, raw_data, times), color in zip(patterns[1:], colors):
            # Convert times to strings for x-axis
            time_labels = [t.strftime('%H:%M') for t in times]
            
            # Add smooth line without markers
            fig.add_trace(go.Scatter(
                x=time_labels,
                y=pattern,
                name=f'Match: {date}',
                line=dict(
                    color=color,
                    width=2,
                    shape='spline',
                    smoothing=1.3
                ),
                mode='lines',
                hovertemplate=f'Time: %{{x}}<br>Change: %{{y:.2f}}%<extra>{date}</extra>'
            ))
            
            # Create proper DataFrame for candlestick data
            candle_times = pd.date_range(
                start=pd.Timestamp('09:30'),
                end=pd.Timestamp('16:00'),
                periods=len(pattern)
            ).time
            
            candle_df = pd.DataFrame({
                'time': [t.strftime('%H:%M') for t in candle_times],
                'open': raw_data['open'],
                'high': raw_data['high'],
                'low': raw_data['low'],
                'close': raw_data['close']
            })
            
            # Add candlestick data with very low opacity
            fig.add_trace(go.Candlestick(
                x=candle_df['time'],
                open=candle_df['open'],
                high=candle_df['high'],
                low=candle_df['low'],
                close=candle_df['close'],
                name=f'Candles {date}',
                opacity=0.15,
                visible='legendonly',
                showlegend=True
            ))
        
        fig.update_layout(
            title='Pattern Comparison (Target vs Similar Historical Patterns)',
            yaxis_title='Percent Change from Open',
            xaxis_title='Market Time',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                font=dict(size=10)
            ),
            margin=dict(r=150),  # More room for legend
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                tickformat='%H:%M',
                tickangle=45,
                tickmode='array',
                ticktext=['09:30', '10:30', '11:30', '12:30', '13:30', '14:30', '15:30', '16:00'],
                tickvals=['09:30', '10:30', '11:30', '12:30', '13:30', '14:30', '15:30', '16:00']
            )
        )
        
        # Update axes for cleaner look
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        
        fig.show()

    def get_patterns_from_db(self, symbol: str, target_date: datetime.date = None) -> list:
        """Get patterns from SQLite database"""
        if not os.path.exists(DB_PATH):
            raise ValueError("Database not found. Please run data_downloader.py first")
            
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            if target_date:
                # Get specific date pattern
                cursor.execute('''
                    SELECT date, pattern_data, time_points, day_open, day_close
                    FROM daily_patterns
                    WHERE symbol = ? AND date = ?
                ''', (symbol, target_date.strftime('%Y-%m-%d')))
            else:
                # Get all patterns
                cursor.execute('''
                    SELECT date, pattern_data, time_points, day_open, day_close
                    FROM daily_patterns
                    WHERE symbol = ?
                ''', (symbol,))
            
            patterns = []
            for row in cursor.fetchall():
                date = datetime.strptime(row[0], '%Y-%m-%d').date()
                pattern = np.frombuffer(row[1], dtype=np.float64)
                
                # Create evenly spaced time points for market hours
                times = pd.date_range(
                    start=pd.Timestamp('09:30'),
                    end=pd.Timestamp('16:00'),
                    periods=len(pattern)
                ).time
                
                day_open = row[3]
                day_close = row[4]
                
                patterns.append({
                    'date': date,
                    'pattern': pattern,
                    'times': times,
                    'day_open': day_open,
                    'day_close': day_close,
                    'final_change': ((day_close - day_open) / day_open) * 100
                })
            
            return patterns

    def calculate_shape_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns"""
        # Ensure patterns are the same length
        min_len = min(len(pattern1), len(pattern2))
        p1 = pattern1[:min_len]
        p2 = pattern2[:min_len]
        
        # Calculate correlation coefficient
        corr = np.corrcoef(p1, p2)[0, 1]
        
        # Calculate mean absolute difference
        diff = np.mean(np.abs(p1 - p2))
        diff_score = 1 / (1 + diff)  # Convert to similarity score (0 to 1)
        
        # Combine scores (70% correlation, 30% difference)
        similarity = 0.7 * max(0, corr) + 0.3 * diff_score
        
        return similarity

    def find_similar_patterns(self, symbol: str, target_date: datetime.date, 
                            start_time: str = "09:30", end_time: str = "16:00",
                            n_matches: int = 5) -> dict:
        """Find patterns similar to the target date's pattern within specified timeframe"""
        # Get target pattern
        target_patterns = self.get_patterns_from_db(symbol, target_date)
        if not target_patterns:
            raise ValueError(f"No pattern found for target date {target_date}")
        
        target_pattern = target_patterns[0]
        
        # Convert time strings to time objects
        start_time = pd.Timestamp(start_time).time()
        end_time = pd.Timestamp(end_time).time()
        
        # Get indices for the specified timeframe
        time_array = pd.date_range(
            start=pd.Timestamp('09:30'),
            end=pd.Timestamp('16:00'),
            periods=len(target_pattern['pattern'])
        ).time
        
        time_mask = [(t >= start_time and t <= end_time) for t in time_array]
        
        # Get the target pattern for the specified timeframe
        target_partial = target_pattern['pattern'][time_mask]
        target_times = [t for t, m in zip(time_array, time_mask) if m]
        
        # Get all patterns
        all_patterns = self.get_patterns_from_db(symbol)
        logger.info(f"Comparing target pattern against {len(all_patterns)} historical patterns")
        logger.info(f"Using timeframe {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}")
        
        # Calculate similarity with all patterns
        similarities = []
        for pattern in all_patterns:
            if pattern['date'] != target_date:
                # Get the same timeframe from the comparison pattern
                pattern_partial = pattern['pattern'][time_mask]
                similarity = self.calculate_shape_similarity(
                    target_partial,
                    pattern_partial
                )
                similarities.append((similarity, pattern, pattern_partial))
        
        # Sort by similarity and get top matches
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_matches = similarities[:n_matches]
        
        # Log results
        logger.info(f"\nTop {n_matches} matching patterns for {target_date}:")
        for similarity, pattern, _ in top_matches:
            logger.info(f"Date: {pattern['date']}, "
                      f"Similarity: {similarity:.2f}, "
                      f"Final Change: {pattern['final_change']:.1f}%")
        
        # Return target timeframe info along with matches
        return {
            'target': {
                'date': target_pattern['date'],
                'pattern': target_partial,
                'times': target_times,
                'full_pattern': target_pattern['pattern'],
                'full_times': time_array,
                'day_open': target_pattern['day_open'],
                'day_close': target_pattern['day_close']
            },
            'matches': [pattern for _, pattern, _ in top_matches]
        }

if __name__ == "__main__":
    load_dotenv()
    
    API_KEY = os.getenv("APCA_API_KEY_ID")
    SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
    
    if not API_KEY or not SECRET_KEY:
        raise ValueError("Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your .env file")
    
    matcher = Chartmatch(API_KEY, SECRET_KEY)
    
    # Convert target date string to datetime
    target_date = pd.Timestamp(TARGET_DATE).date()
    
    try:
        # Find similar patterns using specified timeframe
        results = matcher.find_similar_patterns(
            TARGET_SYMBOL, 
            target_date,
            start_time=TARGET_START_TIME,
            end_time=TARGET_END_TIME,
            n_matches=MATCH_COUNT
        )
        
        # Prepare data for plotting
        plot_data = []
        
        # Add target pattern (show full day pattern)
        target = results['target']
        plot_data.append((
            target['date'],
            target['full_pattern'],  # Use full day pattern instead of partial
            pd.DataFrame({
                'open': np.full(len(target['full_pattern']), target['day_open']),
                'high': np.full(len(target['full_pattern']), target['day_open'] * 1.001),
                'low': np.full(len(target['full_pattern']), target['day_open'] * 0.999),
                'close': np.full(len(target['full_pattern']), target['day_close'])
            }),
            target['full_times']  # Use full day times
        ))
        
        # Add similar patterns (show full day patterns)
        for pattern in results['matches']:
            plot_data.append((
                pattern['date'],
                pattern['pattern'],  # Full day pattern
                pd.DataFrame({
                    'open': np.full(len(pattern['pattern']), pattern['day_open']),
                    'high': np.full(len(pattern['pattern']), pattern['day_open'] * 1.001),
                    'low': np.full(len(pattern['pattern']), pattern['day_open'] * 0.999),
                    'close': np.full(len(pattern['pattern']), pattern['day_close'])
                }),
                pattern['times']  # Full day times
            ))
        
        # Plot patterns
        matcher.plot_patterns(plot_data)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:") 