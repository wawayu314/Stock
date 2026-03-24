# Stock Prediction with Multi-Modal Fusion

A multi-modal stock prediction system combining price features, technical indicators, and news sentiment analysis, designed for multi-market stock prediction.

## Project Overview

This project implements an advanced stock prediction model featuring:

- **Multi-Market Support**: Specialized optimization for NYSE, NASDAQ, SP500, and A-Share markets
- **Multi-Scale Temporal Features**: Multi-scale convolution to capture patterns at different time scales
- **Sparse Industry Attention**: Industry-based sparse attention mechanism for improved computational efficiency
- **News Sentiment Integration**: FinBERT-based sentiment analysis fused with price features via gated multimodal architecture
- **Intelligent Parameter Configuration**: Automatic hyperparameter tuning based on market characteristics

## Key Features

### Multi-Market Configuration System
| Market | Description | Optimization Focus |
|--------|-------------|-------------------|
| NYSE | Mature market | Stability & ranking quality |
| NASDAQ | Tech stock market | Fast-changing dynamics |
| SP500 | Large-cap blue chips | Long-term trends |
| A-Share | Chinese A-shares | High volatility handling |

### Core Innovations

1. **Multi-Scale Time Mixer**: Combines original scale with multiple convolutional scales (stride 2, 4, 8...) via a Triangular Upper (TriU) network for temporal fusion

2. **Sparse Industry Attention**: Reduces O(n²) attention complexity to O(k²×m) by restricting attention to within industry groups

3. **Gated Multimodal Fusion**: Dynamically learns to weight price features vs. sentiment features using a three-layer MLP gate network

4. **Macro-Adaptive Gating**: Incorporates VIX and Treasury Yield signals to adaptively control modality fusion during market stress

## Project Structure

```
Stock/
├── src/
│   ├── model.py                   # Core model: MultiScaleTimeMixer, StockPredict
│   ├── train.py                   # Main training script (price-only)
│   ├── gated_multimodal_train.py  # Gated multimodal fusion training
│   ├── sentiment_lstm_train.py    # Sentiment-enhanced LSTM training
│   ├── news_sentiment_preprocessor.py  # News sentiment preprocessing
│   ├── evaluator.py              # Performance metrics (IC, RIC, Sharpe, etc.)
│   └── load_data.py              # Data loading utilities
├── dataset/                       # Dataset directory (excluded from git)
│   ├── SP500/
│   ├── NASDAQ/
│   └── NYSE/
├── requirements.txt              # Core dependencies
├── requirements_sentiment.txt    # Sentiment analysis dependencies
└── README.md
```

## Installation

### 1. Environment Requirements
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### 2. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Sentiment analysis dependencies (includes FinBERT)
pip install -r requirements_sentiment.txt
```

### 3. Data Preparation

The dataset directory should be organized as follows:
```
dataset/
├── NYSE/
│   ├── eod_data.pkl
│   ├── mask_data.pkl
│   ├── gt_data.pkl
│   ├── price_data.pkl
│   ├── nyse_ticker.csv
│   └── nyse_industry_data.json
├── NASDAQ/
│   ├── eod_data.pkl
│   └── ...
├── SP500/
│   ├── SP500.npy
│   ├── sp500_ticker.csv
│   └── sp500_industry_data.json
│   └── news_sentiment/          # News sentiment data
│       ├── AAPL_daily_sentiment.csv
│       ├── AAPL_news_detailed.csv
│       └── ...
└── A_SHARE/
    └── ...
```

## Usage

### 1. Basic Training (Price Features Only)

```bash
cd src
python train.py
```

### 2. Gated Multimodal Fusion Training (Price + Sentiment)

```bash
cd src
python gated_multimodal_train.py
```

### 3. Switch Markets

Modify the `MARKET_NAME` variable in the training script:

```python
MARKET_NAME = 'SP500'  # Options: 'NYSE', 'NASDAQ', 'SP500', 'A_SHARE'
```

### 4. Custom Parameters

The system automatically selects optimal parameters per market:

| Parameter | NYSE | NASDAQ | SP500 | A_SHARE |
|-----------|------|---------|-------|---------|
| lookback_length | 32 | 64 | 32 | 36 |
| epochs | 60 | 60 | 60 | 150 |
| learning_rate | 5e-5 | 1e-4 | 2e-5 | 8e-5 |
| alpha (rank loss) | 0.75 | 0.8 | 0.5 | 0.2 |
| scale_factor | 3 | 3 | 4 | 1 |

## Performance Metrics

- **IC (Information Coefficient)**: Correlation between predictions and actual returns
- **RIC (Rank Information Coefficient)**: Quality of ranking predictions
- **Precision@10**: Prediction accuracy for top 10% stocks
- **Sharpe Ratio**: Risk-adjusted return metric

## Model Architecture

### Multi-Scale Time Mixer
- Extracts features at multiple temporal scales using 1D convolutions
- Fuses scales via Triangular Upper (TriU) network
- Preserves temporal ordering with causal attention

### Industry Sparse Attention
- Loads SIC-based industry classification
- Restricts attention within industry groups
- 2-5x computational speedup depending on industry distribution

### Gated Multimodal Fusion
- **Price Encoder**: LSTM on price/technical features
- **Sentiment Encoder**: LSTM on news sentiment features
- **Gate Network**: 3-layer MLP predicting modality weights
- **Macro Controller**: VIX + Treasury Yield signal to modulate gating

## Citation

If this work is helpful for your research, please cite:

```
@misc{stock_prediction_2025,
  title = {Stock Prediction with Multi-Modal Fusion},
  author = {Yu Huang},
  year = {2025}
}
```

## License

MIT License

---

**Risk Warning**: This project is for educational and research purposes only. It does not constitute investment advice. Stock investments involve risk; invest with caution.
