# ChartMatch - Real-Time Pattern Matching for Stock Charts

ChartMatch is an experimental Python-based tool that performs real-time pattern matching on stock price charts. It identifies historical patterns similar to the current day's price action and displays them in an interactive dashboard, helping traders identify potential price movement scenarios based on historical data.

## Features

- Real-time pattern matching using minute-by-minute stock data
- Interactive Dash web interface with auto-updating charts
- Configurable similarity thresholds and matching parameters
- Historical pattern database with SQLite storage
- Smooth visualization with Plotly
- Support for any stock symbol available through Alpaca

## Prerequisites

- Python 3.8 or higher
- Alpaca Markets account (with API keys)
- Internet connection for real-time data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/salparadi/chartmatch.git
cd chartmatch
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Alpaca API keys:
```
APCA_API_KEY_ID=your_api_key_here
APCA_API_SECRET_KEY=your_secret_key_here
```

## Usage

1. First, download historical data for your desired symbol:
```bash
python chartmatch_data.py --symbol SPY --days 365
```
This will create a SQLite database (`data/chartmatch.db`) with historical patterns for the specified symbol. Alpaca supports data back to January 1, 2016.

2. Run the live pattern matching dashboard:
```bash
python chartmatch_live.py
```
This will open a web browser with the interactive dashboard showing real-time pattern matching.

## Configuration

Key parameters can be adjusted in the respective Python files:

- `N_MATCHES`: Number of similar patterns to display (default: 10)
- `SIMILARITY_THRESHOLD`: Minimum similarity score for pattern matching (default: 0.7)
- `SMOOTH_WINDOW_SIZE`: Window size for pattern smoothing (default: 21)
- `DASH_UPDATE_MS`: Dashboard update frequency in milliseconds (default: 60000)

## How It Works

1. The system maintains a database of historical price patterns
2. During market hours, it continuously processes incoming price data
3. For each new data point, it:
   - Calculates the current day's pattern
   - Finds similar patterns from the historical database
   - Updates the dashboard with new matches
4. Patterns are compared using multiple factors:
   - Overall shape correlation
   - Recent trend similarity
   - Volatility matching
   - Price movement alignment

## Files

- `chartmatch.py`: Core pattern matching functionality
- `chartmatch_live.py`: Real-time dashboard implementation
- `chartmatch_data.py`: Historical data management and database operations
- `requirements.txt`: Package dependencies
- `.env`: API configuration (you need to create this)
- `data/chartmatch.db`: SQLite database storing historical data and patterns (created automatically)

## Notes

- Running the main script allows you to backtest patterns on historical data
- The live tool works best during market hours (9:30 AM - 4:00 PM ET)
- Historical data download may take several minutes depending on the date range
- Pattern matching accuracy improves with more historical data
- The live dashboard automatically refreshes

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 