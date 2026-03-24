import pickle
import pandas as pd
import os

print("🔍 检查NYSE数据集...")

# 检查NYSE股票数据
dataset_path = '../dataset/NYSE'
eod_file = os.path.join(dataset_path, 'eod_data.pkl')
price_file = os.path.join(dataset_path, 'price_data.pkl')

if os.path.exists(eod_file) and os.path.exists(price_file):
    with open(eod_file, 'rb') as f:
        eod_data = pickle.load(f)
    with open(price_file, 'rb') as f:
        price_data = pickle.load(f)
    
    print(f"📊 NYSE股票数据:")
    print(f"   EOD数据形状: {eod_data.shape}")
    print(f"   价格数据形状: {price_data.shape}")
    print(f"   股票数量: {eod_data.shape[0]}")
    print(f"   交易天数: {eod_data.shape[1]}")
    print(f"   特征数量: {eod_data.shape[2]}")
else:
    print("❌ NYSE pickle文件不存在")

# 检查NYSE情绪数据
sentiment_file = '../dataset/processed_sentiment/nyse_processed_sentiment.csv'
if os.path.exists(sentiment_file):
    sentiment_df = pd.read_csv(sentiment_file)
    print(f"\n📰 NYSE情绪数据:")
    print(f"   总记录数: {len(sentiment_df)}")
    print(f"   股票数量: {len(sentiment_df['ticker'].unique())}")
    print(f"   股票代码: {sorted(sentiment_df['ticker'].unique())}")
    
    # 检查每只股票的数据量
    ticker_counts = sentiment_df['ticker'].value_counts()
    print(f"\n📈 各股票情绪数据数量:")
    for ticker, count in ticker_counts.items():
        print(f"   {ticker}: {count} 条记录")
    
    # 检查时间范围
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    print(f"\n📅 时间范围:")
    print(f"   开始日期: {sentiment_df['date'].min().strftime('%Y-%m-%d')}")
    print(f"   结束日期: {sentiment_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   总天数: {(sentiment_df['date'].max() - sentiment_df['date'].min()).days + 1}")
else:
    print("❌ NYSE情绪数据文件不存在")

print("\n💡 建议:")
print("1. 如果NYSE股票数据有多只股票，但情绪数据只有少数几只")
print("2. 可以考虑:")
print("   - 扩展情绪数据收集到更多NYSE股票")
print("   - 或者使用NASDAQ数据集（情绪数据更完整）")
print("   - 或者为没有情绪数据的股票使用平均情绪或零填充") 