import pandas as pd
import numpy as np
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
import plotly.graph_objects as go
from datetime import datetime
from scipy.signal import savgol_filter
import os
from dotenv import load_dotenv
import logging
import sqlite3
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import threading
import webbrowser
from chartmatch import Chartmatch, SMOOTH_WINDOW, DB_PATH

# Configuration Constants
N_MATCHES = 10                   # Number of similar patterns to find and display
DASH_UPDATE_MS = 5000            # Dash update frequency in milliseconds
SIMILARITY_THRESHOLD = 0.7       # Minimum similarity score to consider a match
SMOOTH_WINDOW_SIZE = 21          # Window size for Savitzky-Golay smoothing
RECENT_WEIGHT = 0.6              # Weight given to recent pattern segments in similarity calculation
SPLINE_SMOOTHING = 1.3           # Smoothing factor for plotly spline curves
CHART_LINE_WIDTH = 3             # Width of the current pattern line
MATCH_LINE_WIDTH = 2             # Width of the matching patterns lines

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChartmatchLive(Chartmatch):
    def __init__(self, api_key: str, secret_key: str):
        """Initialize with Alpaca credentials and setup websocket"""
        super().__init__(api_key, secret_key)
        self.current_bars = []
        self.fig = None
        self.current_day_open = None
        self.symbol = None
        self.api_key = api_key
        self.secret_key = secret_key
        self.stock_data_stream_client = None
        self.app = None
        self.setup_dash()
        
    def setup_dash(self):
        """Initialize Dash app and layout"""
        self.app = Dash(__name__)
        
        # Add CSS for full height
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>ChartMatch Live</title>
                {%favicon%}
                {%css%}
                <style>
                    html, body {
                        height: 100%;
                        margin: 0;
                        padding: 0;
                    }
                    #react-entry-point {
                        height: 100%;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        self.app.layout = html.Div([
            html.Div([
                dcc.Graph(
                    id='live-chart',
                    style={'height': 'calc(100vh)'}  # Full height minus header
                )
            ], style={'height': 'calc(100%)'}),
            dcc.Interval(
                id='interval-component',
                interval=DASH_UPDATE_MS,  # Use constant instead of hardcoded value
                n_intervals=0
            )
        ], style={'height': '100vh'})
        
        @self.app.callback(
            Output('live-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_graph(n):
            if self.fig is None:
                self.initialize_plot()
            return self.fig
        
    def start_dash(self, port=8050):
        """Start Dash server and open browser"""
        # Open browser automatically
        webbrowser.open(f'http://localhost:{port}')
        
        # Start the server
        self.app.run_server(debug=False, port=port)
        
    async def handle_bar(self, bar):
        """Async handler for incoming bar data"""
        logger.info(f"Received bar: {bar}")
        self.process_bar(bar)
        
    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical bar data from Alpaca"""
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = pd.Timestamp(start_date, tz='America/New_York')
        if end_date.tzinfo is None:
            end_date = pd.Timestamp(end_date, tz='America/New_York')
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date,
            feed=DataFeed.SIP
        )
        
        try:
            bars = self.data_client.get_stock_bars(request)
            
            if not hasattr(bars, 'data') or symbol not in bars.data:
                logger.warning(f"No data returned from Alpaca for {symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars.data[symbol]])
            
            if df.empty:
                logger.warning("DataFrame is empty after processing bars")
                return df
                
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index).tz_convert('America/New_York')
            
            logger.info(f"Fetched {len(df)} bars")
            if not df.empty:
                logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
        
    def start_live_feed(self, symbol: str):
        """Start websocket connection and subscribe to symbol"""
        self.symbol = symbol
        
        # First, verify we have the database with historical patterns
        if not os.path.exists(DB_PATH):
            raise ValueError("Database not found. Please run data_downloader.py first")
        
        try:
            # Get today's bars so far
            now = datetime.now()
            today = now.date()
            
            # Calculate market open time for today
            market_open = datetime.combine(today, pd.Timestamp('09:30').time())
            # Use current time for end
            current_time = now
            
            # If we're before market open, don't try to fetch data
            if now < market_open:
                logger.info("Market not open yet")
                self.current_bars = []
            else:
                logger.info(f"Fetching today's bars for {symbol} from {market_open} to {current_time}")
                bars_df = self.fetch_historical_data(symbol, market_open, current_time)
                
                if bars_df.empty:
                    logger.warning("No historical bars found for today")
                else:
                    # Initialize current_bars with historical data
                    self.current_bars = [{
                        'timestamp': idx,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    } for idx, row in bars_df.iterrows()]
                    
                    if self.current_bars:
                        self.current_day_open = self.current_bars[0]['open']
                        logger.info(f"Loaded {len(self.current_bars)} bars for today")
                        logger.info(f"First bar time: {self.current_bars[0]['timestamp']}")
                        logger.info(f"Last bar time: {self.current_bars[-1]['timestamp']}")
                        
                        # Do initial pattern match with what we have
                        self.update_analysis()
            
        except Exception as e:
            logger.error(f"Error fetching today's data: {e}")
            logger.exception("Full traceback:")
            logger.info("Continuing with empty data...")
        
        # Create websocket client
        self.stock_data_stream_client = StockDataStream(
            feed=DataFeed.SIP,
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        # Subscribe to bars
        self.stock_data_stream_client.subscribe_bars(self.handle_bar, symbol)
        
        # Start the websocket in a separate thread
        websocket_thread = threading.Thread(target=self.stock_data_stream_client.run)
        websocket_thread.daemon = True
        websocket_thread.start()
        
        # Start Dash server
        self.start_dash()
        
    def process_bar(self, bar):
        """Process the incoming bar data"""
        # Skip if not market hours
        bar_time = pd.Timestamp(bar.timestamp).tz_convert('America/New_York')
        if not self.is_market_hours(bar_time):
            return
            
        # Initialize day_open if needed
        if self.current_day_open is None:
            self.current_day_open = bar.open
            
        # Check if this is a new bar we haven't seen
        if not any(b['timestamp'] == bar_time for b in self.current_bars):
            # Store bar
            self.current_bars.append({
                'timestamp': bar_time,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
            
            # Sort bars by timestamp to maintain order
            self.current_bars.sort(key=lambda x: x['timestamp'])
            
            # Update pattern matching and plot
            try:
                self.update_analysis()
            except Exception as e:
                logger.error(f"Error updating analysis: {e}")
        
    def is_market_hours(self, timestamp):
        """Check if timestamp is during market hours"""
        time = timestamp.time()
        return (
            time >= pd.Timestamp('09:30').time() and 
            time <= pd.Timestamp('16:00').time()
        )
        
    def calculate_shape_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns"""
        # Ensure patterns are the same length
        min_len = min(len(pattern1), len(pattern2))
        if min_len < 3:  # Need at least 3 points for meaningful comparison
            return 0.0
            
        p1 = pattern1[:min_len]
        p2 = pattern2[:min_len]
        
        try:
            # Calculate correlation coefficient (shape similarity)
            corr = np.corrcoef(p1, p2)[0, 1]
            if np.isnan(corr):
                return 0.0
            
            # Calculate recent trend similarity (last RECENT_WEIGHT of the pattern)
            recent_idx = int((1 - RECENT_WEIGHT) * min_len)
            p1_recent = p1[recent_idx:]
            p2_recent = p2[recent_idx:]
            recent_corr = np.corrcoef(p1_recent, p2_recent)[0, 1]
            if np.isnan(recent_corr):
                recent_corr = 0.0
            
            # Calculate mean absolute difference with more weight on recent values
            full_diff = np.mean(np.abs(p1 - p2))
            recent_diff = np.mean(np.abs(p1_recent - p2_recent))
            diff_score = 1 / (1 + (0.4 * full_diff + 0.6 * recent_diff))
            
            # Calculate volatility similarity
            vol1 = np.std(np.diff(p1))
            vol2 = np.std(np.diff(p2))
            vol_sim = 1 - min(abs(vol1 - vol2) / max(vol1, vol2, 0.001), 1)
            
            # Calculate turning points similarity
            turns1 = len(np.where(np.diff(np.sign(np.diff(p1))))[0])
            turns2 = len(np.where(np.diff(np.sign(np.diff(p2))))[0])
            turns_sim = 1 - min(abs(turns1 - turns2) / max(turns1, turns2, 1), 1)
            
            # Combine scores with higher weight on correlation and recent pattern
            similarity = (
                0.35 * max(0, corr) +          # Overall shape correlation
                0.25 * max(0, recent_corr) +   # Recent shape correlation
                0.20 * diff_score +            # Price difference (weighted recent)
                0.10 * vol_sim +               # Volatility similarity
                0.10 * turns_sim               # Turning points similarity
            )
            
            logger.debug(f"Similarity components - Corr: {corr:.3f}, Recent: {recent_corr:.3f}, "
                        f"Diff: {diff_score:.3f}, Vol: {vol_sim:.3f}, Turns: {turns_sim:.3f}")
            
            # Apply stricter threshold
            return similarity if similarity > SIMILARITY_THRESHOLD else 0.0
            
        except Exception as e:
            logger.debug(f"Error in similarity calculation: {e}")
            return 0.0
        
    def find_matches_for_live_pattern(self, current_pattern: np.ndarray, current_times: list, n_matches: int = 5) -> dict:
        """Find historical patterns similar to current live pattern"""
        if not os.path.exists(DB_PATH):
            raise ValueError("Database not found. Please run data_downloader.py first")
            
        # Need at least a few bars to make a meaningful comparison
        if len(current_pattern) < 3:
            logger.info("Need at least 3 bars to find matches")
            return {
                'target': {
                    'date': datetime.now().date(),
                    'pattern': current_pattern,
                    'times': current_times,
                    'day_open': self.current_day_open
                },
                'matches': []
            }
            
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Get all historical patterns (excluding today)
            today = datetime.now().date()
            cursor.execute('''
                SELECT date, pattern_data, time_points, day_open, day_close
                FROM daily_patterns
                WHERE symbol = ? AND date < ?
            ''', (self.symbol, today.strftime('%Y-%m-%d')))
            
            patterns = []
            similarities = []
            total_patterns = 0
            patterns_checked = 0
            
            # Calculate minutes since market open for current pattern
            first_time = pd.Timestamp.combine(today, current_times[0])
            last_time = pd.Timestamp.combine(today, current_times[-1])
            minutes_elapsed = int((last_time - first_time).total_seconds() / 60) + 1
            
            for row in cursor.fetchall():
                total_patterns += 1
                date = datetime.strptime(row[0], '%Y-%m-%d').date()
                hist_pattern = np.frombuffer(row[1], dtype=np.float64)
                
                # Create minute-by-minute pattern for the historical data
                market_open = pd.Timestamp.combine(date, pd.Timestamp('09:30').time())
                market_close = pd.Timestamp.combine(date, pd.Timestamp('16:00').time())
                full_day_times = pd.date_range(start=market_open, end=market_close, freq='1min')
                
                # Interpolate historical pattern to minute-by-minute
                full_index = np.arange(len(full_day_times))
                hist_index = np.linspace(0, len(full_day_times)-1, len(hist_pattern))
                hist_minute = np.interp(full_index, hist_index, hist_pattern)
                
                # Get matching segment (same number of minutes as current pattern)
                if len(hist_minute) < minutes_elapsed:
                    logger.debug(f"Skipping {date} - insufficient points")
                    continue
                    
                hist_segment = hist_minute[:minutes_elapsed]
                patterns_checked += 1
                
                try:
                    # Calculate similarity
                    similarity = self.calculate_shape_similarity(current_pattern, hist_segment)
                    
                    # Only include if similarity is meaningful
                    if not np.isnan(similarity) and similarity > SIMILARITY_THRESHOLD:
                        day_open = row[3]
                        day_close = row[4]
                        
                        pattern_data = {
                            'date': date,
                            'pattern': hist_minute,  # Full minute-by-minute pattern
                            'matching_segment': hist_segment,  # Just the matching part
                            'day_open': day_open,
                            'day_close': day_close,
                            'final_change': ((day_close - day_open) / day_open) * 100,
                            'similarity': similarity
                        }
                        
                        similarities.append((similarity, pattern_data))
                    else:
                        logger.debug(f"Pattern from {date} similarity too low: {similarity:.3f}")
                except Exception as e:
                    logger.debug(f"Error calculating similarity for {date}: {e}")
                    continue
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_matches = similarities[:n_matches]
            
            # Log results
            logger.info(f"\nPattern matching stats:")
            logger.info(f"Total patterns in DB: {total_patterns}")
            logger.info(f"Patterns checked: {patterns_checked}")
            logger.info(f"Matches found: {len(similarities)}")
            logger.info(f"\nTop {len(top_matches)} matching patterns:")
            for similarity, pattern in top_matches:
                logger.info(f"Date: {pattern['date']}, "
                          f"Similarity: {similarity:.2f}, "
                          f"Final Change: {pattern['final_change']:.1f}%")
            
            return {
                'target': {
                    'date': today,
                    'pattern': current_pattern,
                    'times': current_times,
                    'day_open': self.current_day_open
                },
                'matches': [p for _, p in top_matches]
            }
    
    def update_analysis(self):
        """Update pattern analysis with latest data"""
        if not self.current_bars:
            return
            
        # Convert bars to DataFrame
        df = pd.DataFrame(self.current_bars)
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Preparing pattern from {len(df)} bars")
        logger.info(f"Time range: {df.index[0]} to {df.index[-1]}")
        
        # Prepare current pattern
        pattern, times = self.prepare_day_pattern(df)
        
        logger.info(f"Generated pattern with {len(pattern)} points")
        
        # Find similar patterns from database
        results = self.find_matches_for_live_pattern(pattern, times, n_matches=N_MATCHES)
        
        # Update plot
        self.update_plot(results)
        
    def initialize_plot(self):
        """Create initial plotly figure"""
        self.fig = go.Figure()
        
        # Set up layout
        self.fig.update_layout(
            title=f'ChartMatch Live Pattern Comparison for {self.symbol}',
            yaxis_title='Percent Change from Open',
            xaxis_title='Market Time',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            xaxis=dict(
                tickformat='%H:%M',
                tickangle=45,
                tickmode='array',
                ticktext=['09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', 
                         '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00'],
                tickvals=['09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', 
                         '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00'],
                gridcolor='#f0f0f0'
            ),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                gridcolor='#f0f0f0'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                font=dict(size=10)
            ),
            margin=dict(r=150)  # More room for legend
        )
        
        # Initialize with empty trace for current pattern
        self.fig.add_trace(go.Scatter(
            x=[],
            y=[],
            name='Current Pattern',
            line=dict(
                color='black',
                width=CHART_LINE_WIDTH,
                shape='spline',
                smoothing=SPLINE_SMOOTHING
            ),
            mode='lines'
        ))
        
    def update_plot(self, results):
        """Update plot with new data"""
        if self.fig is None:
            self.initialize_plot()
            
        # Clear all traces
        self.fig.data = []
        
        # Get current pattern info
        current_pattern = results['target']
        
        # Calculate minutes since market open using the actual timestamp from current_bars
        market_open = pd.Timestamp('today').replace(hour=9, minute=30).tz_localize('America/New_York')
        last_bar_time = pd.Timestamp(self.current_bars[-1]['timestamp'])
        if last_bar_time.tzinfo is None:
            last_bar_time = last_bar_time.tz_localize('America/New_York')
        else:
            last_bar_time = last_bar_time.tz_convert('America/New_York')
            
        minutes_elapsed = int((last_bar_time - market_open).total_seconds() / 60)
        
        # Create time points for full trading day (390 minutes)
        full_day_times = pd.date_range(
            start=market_open,
            periods=390,
            freq='1min'
        )
        time_labels = [t.strftime('%H:%M') for t in full_day_times]
        
        # Add current pattern (only up to current time)
        self.fig.add_trace(go.Scatter(
            x=time_labels[:minutes_elapsed+1],
            y=current_pattern['pattern'],
            name='Current Pattern',
            line=dict(
                color='black',
                width=CHART_LINE_WIDTH,
                shape='spline',
                smoothing=SPLINE_SMOOTHING
            ),
            mode='lines'
        ))
        
        # Add similar patterns
        colors = [
            '#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
            '#17becf', '#e377c2', '#bcbd22', '#7f7f7f', '#8c564b'
        ]
        
        for pattern, color in zip(results['matches'], colors):
            # Apply smoothing to matching segment
            matching_segment = pattern['matching_segment']
            if len(matching_segment) > 3:
                matching_segment = savgol_filter(matching_segment, 
                                              min(21, len(matching_segment) - (1 if len(matching_segment) % 2 == 0 else 0)), 
                                              3)
            
            # Plot matching segment (solid line)
            self.fig.add_trace(go.Scatter(
                x=time_labels[:minutes_elapsed+1],
                y=matching_segment,
                name=f"Match: {pattern['date']} (Final: {pattern['final_change']:.1f}%)",
                line=dict(
                    color=color,
                    width=MATCH_LINE_WIDTH,
                    shape='spline',
                    smoothing=SPLINE_SMOOTHING
                ),
                mode='lines'
            ))
            
            # Interpolate remainder of the day
            if len(pattern['pattern']) > len(pattern['matching_segment']):
                remainder_start = len(pattern['matching_segment']) - 1  # Overlap one point
                remainder = pattern['pattern'][remainder_start:]
                
                # Apply Savitzky-Golay smoothing to remainder
                if len(remainder) > 3:
                    remainder = savgol_filter(remainder, 
                                           min(21, len(remainder) - (1 if len(remainder) % 2 == 0 else 0)), 
                                           3)
                
                remainder_times = time_labels[minutes_elapsed:minutes_elapsed + len(remainder)]
                
                # Plot remainder (dashed)
                self.fig.add_trace(go.Scatter(
                    x=remainder_times,
                    y=remainder,
                    name=f"Match: {pattern['date']} (Remainder)",
                    line=dict(
                        color=color,
                        width=MATCH_LINE_WIDTH,
                        dash='dash',
                        shape='spline',
                        smoothing=SPLINE_SMOOTHING
                    ),
                    mode='lines',
                    showlegend=False
                ))
        
        # Update layout
        self.fig.update_layout(
            title=f'ChartMatch Live Pattern Comparison for {self.symbol}',
            yaxis_title='Percent Change from Open',
            xaxis_title='Market Time',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            xaxis=dict(
                tickformat='%H:%M',
                tickangle=45,
                tickmode='array',
                ticktext=['09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', 
                         '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00'],
                tickvals=['09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', 
                         '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00'],
                gridcolor='#f0f0f0',
                range=[time_labels[0], time_labels[-1]]  # Always show full day
            ),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                gridcolor='#f0f0f0',
                autorange=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                font=dict(size=10)
            ),
            margin=dict(r=150),  # More room for legend
            shapes=[
                # Add vertical line at current time
                dict(
                    type='line',
                    x0=time_labels[minutes_elapsed],
                    x1=time_labels[minutes_elapsed],
                    y0=0,
                    y1=1,
                    yref='paper',
                    line=dict(
                        color='gray',
                        width=1,
                        dash='dash'
                    )
                )
            ]
        )

    def prepare_day_pattern(self, day_data: pd.DataFrame, smooth_window: int = SMOOTH_WINDOW, target_points: int = None) -> tuple:
        """Create smoothed HLC3 pattern for a single day"""
        # Calculate HLC3
        hlc3 = (day_data['high'] + day_data['low'] + day_data['close']) / 3
        
        # Convert to percentage change from day's open
        day_open = day_data['open'].iloc[0]
        pct_change = (hlc3 - day_open) / day_open * 100
        
        logger.debug(f"Raw pattern length: {len(pct_change)}")
        
        # First apply smoothing with larger window
        if len(pct_change) >= smooth_window >= 3:
            smoothed = savgol_filter(pct_change, smooth_window, 3)
            logger.debug("Applied smoothing filter")
        else:
            smoothed = pct_change
            logger.debug("Skipped smoothing (not enough points)")
        
        # Create time points for market hours (one per minute)
        market_open = day_data.index[0].replace(hour=9, minute=30)
        time_points = pd.date_range(
            start=market_open,
            end=day_data.index[-1],
            freq='1min'
        )
        
        # Interpolate to minute-by-minute data points
        minute_index = np.arange(len(time_points))
        data_index = np.linspace(0, len(minute_index)-1, len(smoothed))
        smoothed = np.interp(minute_index, data_index, smoothed)
        
        return smoothed, time_points.time

def main():
    load_dotenv()
    
    API_KEY = os.getenv("APCA_API_KEY_ID")
    SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
    
    if not API_KEY or not SECRET_KEY:
        raise ValueError("Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your .env file")
    
    matcher = ChartmatchLive(API_KEY, SECRET_KEY)
    
    try:
        matcher.start_live_feed('SPY')
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == "__main__":
    main() 