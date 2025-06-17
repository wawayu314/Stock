import requests
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import pickle
import re
from textblob import TextBlob
import json

# Polygon.io API配置 - 升级版：无限制调用 + 新闻情绪分析
POLYGON_API_KEY = "NoJWwFDrkW3bh0oyp1Wc8ZhYjLu_0zZX"  # 用户的升级版API密钥
POLYGON_BASE_URL = "https://api.polygon.io"

def get_news_for_ticker(ticker, start_date, end_date, limit=1000):
    """获取特定股票的新闻数据"""
    try:
        url = f"{POLYGON_BASE_URL}/v2/reference/news"
        params = {
            'ticker': ticker,
            'published_utc.gte': start_date,
            'published_utc.lte': end_date,
            'order': 'asc',
            'sort': 'published_utc',
            'limit': limit,
            'apikey': POLYGON_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 429:
            print(f"速率限制，等待5秒... ({ticker})")
            time.sleep(5)
            response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"新闻API请求失败: {response.status_code} - {ticker}")
            return None
            
        data = response.json()
        
        if 'results' not in data or not data['results']:
            print(f"无新闻数据: {ticker}")
            return None
        
        return data['results']
        
    except Exception as e:
        print(f"获取新闻数据失败: {ticker}, 错误: {e}")
        return None

def analyze_sentiment_textblob(text):
    """使用TextBlob进行基础情绪分析"""
    try:
        blob = TextBlob(text)
        # 情绪极性: -1 (负面) 到 1 (正面)
        polarity = blob.sentiment.polarity
        # 主观性: 0 (客观) 到 1 (主观)
        subjectivity = blob.sentiment.subjectivity
        
        # 分类情绪
        if polarity > 0.1:
            sentiment_label = 'positive'
        elif polarity < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
            
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment_label': sentiment_label,
            'confidence': abs(polarity)
        }
    except:
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.0
        }

def analyze_sentiment_advanced(text):
    """高级情绪分析 - 可以集成更先进的模型"""
    # 这里可以集成如 VADER, BERT, FinBERT 等更先进的情绪分析模型
    # 现在先使用基础的TextBlob
    return analyze_sentiment_textblob(text)

def extract_key_features(news_item):
    """从新闻中提取关键特征"""
    try:
        title = news_item.get('title', '')
        description = news_item.get('description', '')
        full_text = f"{title}. {description}"
        
        # 情绪分析
        sentiment = analyze_sentiment_advanced(full_text)
        
        # 关键词检测
        keywords = {
            'earnings': any(word in full_text.lower() for word in ['earning', 'revenue', 'profit', 'loss']),
            'acquisition': any(word in full_text.lower() for word in ['acquisition', 'merger', 'acquire', 'buy']),
            'regulation': any(word in full_text.lower() for word in ['regulation', 'regulatory', 'sec', 'fda']),
            'partnership': any(word in full_text.lower() for word in ['partnership', 'collaborate', 'joint']),
            'product': any(word in full_text.lower() for word in ['product', 'launch', 'release', 'innovation']),
            'financial': any(word in full_text.lower() for word in ['dividend', 'split', 'buyback', 'debt']),
            'leadership': any(word in full_text.lower() for word in ['ceo', 'cfo', 'management', 'director']),
            'lawsuit': any(word in full_text.lower() for word in ['lawsuit', 'legal', 'court', 'settlement'])
        }
        
        # 紧急程度评估
        urgency_words = ['breaking', 'urgent', 'alert', 'important', 'critical']
        urgency_score = sum(1 for word in urgency_words if word in full_text.lower()) / len(urgency_words)
        
        return {
            'title': title,
            'description': description,
            'published_utc': news_item.get('published_utc'),
            'article_url': news_item.get('article_url'),
            'publisher': news_item.get('publisher', {}).get('name', ''),
            'sentiment_polarity': sentiment['polarity'],
            'sentiment_subjectivity': sentiment['subjectivity'],
            'sentiment_label': sentiment['sentiment_label'],
            'sentiment_confidence': sentiment['confidence'],
            'keywords_earnings': keywords['earnings'],
            'keywords_acquisition': keywords['acquisition'],
            'keywords_regulation': keywords['regulation'],
            'keywords_partnership': keywords['partnership'],
            'keywords_product': keywords['product'],
            'keywords_financial': keywords['financial'],
            'keywords_leadership': keywords['leadership'],
            'keywords_lawsuit': keywords['lawsuit'],
            'urgency_score': urgency_score,
            'text_length': len(full_text),
            'tickers': news_item.get('tickers', [])
        }
    except Exception as e:
        print(f"特征提取错误: {e}")
        return None

def process_ticker_news(ticker, start_date, end_date, news_dir):
    """处理单个股票的新闻数据"""
    print(f"正在处理 {ticker} 的新闻数据...")
    
    # 获取新闻数据
    news_data = get_news_for_ticker(ticker, start_date, end_date)
    
    if not news_data:
        return None
    
    # 处理每条新闻
    processed_news = []
    for news_item in news_data:
        features = extract_key_features(news_item)
        if features:
            features['ticker'] = ticker
            processed_news.append(features)
    
    if not processed_news:
        return None
    
    # 创建DataFrame
    df = pd.DataFrame(processed_news)
    
    # 转换时间格式
    df['published_utc'] = pd.to_datetime(df['published_utc'])
    df['date'] = df['published_utc'].dt.date
    
    # 按日期聚合新闻情绪
    daily_sentiment = df.groupby('date').agg({
        'sentiment_polarity': ['mean', 'std', 'count'],
        'sentiment_confidence': 'mean',
        'urgency_score': 'mean',
        'keywords_earnings': 'sum',
        'keywords_acquisition': 'sum',
        'keywords_regulation': 'sum',
        'keywords_partnership': 'sum',
        'keywords_product': 'sum',
        'keywords_financial': 'sum',
        'keywords_leadership': 'sum',
        'keywords_lawsuit': 'sum',
        'text_length': 'mean'
    }).round(4)
    
    # 重命名列
    daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns]
    daily_sentiment = daily_sentiment.reset_index()
    daily_sentiment['ticker'] = ticker
    
    # 保存详细新闻数据
    news_filename = f"{ticker}_news_detailed.csv"
    news_filepath = os.path.join(news_dir, news_filename)
    df.to_csv(news_filepath, index=False)
    
    # 保存日度聚合情绪数据
    sentiment_filename = f"{ticker}_daily_sentiment.csv"
    sentiment_filepath = os.path.join(news_dir, sentiment_filename)
    daily_sentiment.to_csv(sentiment_filepath, index=False)
    
    print(f"✓ {ticker}: {len(processed_news)} 条新闻, {len(daily_sentiment)} 交易日")
    
    return daily_sentiment

def get_sp500_tickers():
    """获取S&P 500股票列表 - 从数据集文件中读取"""
    # 读取数据集中的SP500股票列表
    OUTPUT_BASE_DIR = '../dataset'
    sp500_file_path = os.path.join(OUTPUT_BASE_DIR, 'SP500', 'sp500_ticker.csv')
    
    if not os.path.exists(sp500_file_path):
        print(f"错误：未找到S&P 500股票列表文件 {sp500_file_path}")
        return []
    
    try:
        # 读取CSV文件
        df = pd.read_csv(sp500_file_path)
        sp500_tickers = df['Symbol'].tolist()
        
        print(f"从数据集文件读取到 {len(sp500_tickers)} 只S&P 500股票")
        return sp500_tickers
        
    except Exception as e:
        print(f"读取S&P 500数据文件出错: {e}")
        return []

def download_news_batch(ticker_list, start_date, end_date, news_dir, batch_size=200):
    """批量下载新闻情绪数据 - 升级版API优化"""
    print(f"\n{'='*60}")
    print(f"开始下载S&P 500新闻情绪数据 (升级版API - 高速模式)")
    print(f"股票数量: {len(ticker_list)}")
    print(f"时间范围: {start_date} 到 {end_date}")
    print(f"批次大小: {batch_size} (升级版优化)")
    print(f"{'='*60}")
    
    all_daily_sentiment = []
    successful_tickers = []
    
    # 分批处理，每批200只股票 (升级版API支持更大批次)
    for batch_start in range(0, len(ticker_list), batch_size):
        batch_end = min(batch_start + batch_size, len(ticker_list))
        batch_tickers = ticker_list[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(ticker_list) + batch_size - 1) // batch_size
        
        print(f"\n处理第 {batch_num}/{total_batches} 批 ({len(batch_tickers)} 只股票)")
        
        for ticker in tqdm(batch_tickers, desc=f"批次 {batch_num}"):
            daily_sentiment = process_ticker_news(ticker, start_date, end_date, news_dir)
            
            if daily_sentiment is not None:
                all_daily_sentiment.append(daily_sentiment)
                successful_tickers.append(ticker)
            
            # 升级版API - 大幅减少请求间隔
            time.sleep(0.05)  # 从0.3秒减少到0.05秒
        
        # 升级版API - 减少批次间休息时间
        if batch_num < total_batches:
            print(f"批次 {batch_num} 完成，休息1秒后继续...")  # 从10秒减少到1秒
            time.sleep(1)
    
    # 合并所有日度情绪数据
    if all_daily_sentiment:
        combined_sentiment = pd.concat(all_daily_sentiment, ignore_index=True)
        combined_filepath = os.path.join(news_dir, 'sp500_combined_daily_sentiment.csv')
        combined_sentiment.to_csv(combined_filepath, index=False)
        
        print(f"\n✅ S&P 500新闻数据收集完成！(升级版API高速模式)")
        print(f"成功处理 {len(successful_tickers)} 只股票的新闻数据")
        print(f"合并文件已保存: {combined_filepath}")
        
        return combined_sentiment, successful_tickers
    else:
        print("没有成功获取到任何新闻数据")
        return None, []

def main():
    """主执行函数"""
    # --- 配置信息 ---
    OUTPUT_BASE_DIR = '../dataset'
    MARKET_NAME = 'SP500'
    DATA_DIR = os.path.join(OUTPUT_BASE_DIR, MARKET_NAME)
    NEWS_DIR = os.path.join(DATA_DIR, 'news_sentiment')
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(NEWS_DIR, exist_ok=True)
    
    # 设置时间范围 - 与股价数据对齐
    START_DATE_STR = '2020-01-01'
    END_DATE_STR = '2024-12-31'
    
    print(f"📈 S&P 500新闻情绪数据收集 (升级版API - 高速模式)")
    print(f"新闻情绪数据将保存到: {os.path.abspath(NEWS_DIR)}")
    print(f"数据时间范围: {START_DATE_STR} 到 {END_DATE_STR}")
    print(f"✅ 升级版API：无限制调用 + 高速收集优化")

    # 1. 获取S&P 500股票列表
    sp500_tickers = get_sp500_tickers()
    
    if not sp500_tickers:
        print("错误：无法获取S&P 500股票列表")
        return
    
    # 保存股票列表
    ticker_list_path = os.path.join(DATA_DIR, 'sp500_tickers.txt')
    with open(ticker_list_path, 'w') as f:
        for ticker in sp500_tickers:
            f.write(f"{ticker}\n")
    
    print(f"📊 数据集统计:")
    print(f"  - S&P 500股票总数: {len(sp500_tickers)} 只")
    print(f"  - 收集模式: 全量收集 (数据集中所有股票)")
    print(f"  - API优化: 升级版高速模式")
    print(f"  - 股票列表已保存: {ticker_list_path}")
    
    # 显示前10只和最后10只股票作为预览
    print(f"\n📝 股票列表预览:")
    print(f"前10只: {', '.join(sp500_tickers[:10])}")
    print(f"最后10只: {', '.join(sp500_tickers[-10:])}")
    
    confirm = input(f"\n确认开始收集全部 {len(sp500_tickers)} 只S&P 500股票的新闻数据？(y/n): ").strip().lower()
    if confirm != 'y':
        print("取消收集")
        return
    
    process_tickers = sp500_tickers

    # 2. 下载新闻情绪数据
    print(f"\n🔥 开始高速收集模式...")
    combined_sentiment, successful_tickers = download_news_batch(
        process_tickers, START_DATE_STR, END_DATE_STR, NEWS_DIR
    )
    
    if combined_sentiment is not None:
        # 3. 保存成功处理的股票列表
        success_ticker_path = os.path.join(NEWS_DIR, 'sp500_news_tickers.txt')
        with open(success_ticker_path, 'w') as f:
            for ticker in successful_tickers:
                f.write(f"{ticker}\n")
        
        print(f"\n📊 S&P 500新闻数据收集总结 (升级版高速模式):")
        print(f"  - 数据集总股票数: {len(sp500_tickers)}")
        print(f"  - 成功处理股票数: {len(successful_tickers)}")
        print(f"  - 成功率: {len(successful_tickers)/len(sp500_tickers)*100:.1f}%")
        print(f"  - 数据保存位置: {NEWS_DIR}")
        print(f"  - 股票列表保存: {success_ticker_path}")
        print(f"  - 合并数据文件: sp500_combined_daily_sentiment.csv")
        print(f"  - 收集效率: 升级版API高速优化")
        
    else:
        print("❌ S&P 500新闻情绪数据收集失败")

if __name__ == '__main__':
    main() 