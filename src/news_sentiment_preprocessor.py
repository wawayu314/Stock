import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import time

class NewsDataPreprocessor:
    """新闻数据预处理器 - 处理多市场新闻数据"""
    
    def __init__(self, config):
        self.config = config
        self.markets = ['NASDAQ', 'SP500', 'NYSE', 'A_SHARE']
        
        # 初始化情绪分析工具
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # 初始化FinBERT模型，添加重试机制
        self.finbert_tokenizer = None
        self.finbert_model = None
        self._initialize_finbert()
        
        # 新闻类型关键词字典
        self.news_type_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'financial results', 'quarterly'],
            'merger': ['merger', 'acquisition', 'takeover', 'buyout', 'deal'],
            'regulation': ['regulation', 'regulatory', 'compliance', 'SEC', 'FDA'],
            'partnership': ['partnership', 'collaboration', 'alliance', 'joint venture'],
            'product': ['product', 'launch', 'innovation', 'technology', 'patent'],
            'leadership': ['CEO', 'CFO', 'management', 'executive', 'leadership'],
            'lawsuit': ['lawsuit', 'litigation', 'legal', 'court', 'settlement'],
            'market': ['market', 'industry', 'sector', 'competition', 'share price']
        }
        
    def _initialize_finbert(self, max_retries=3):
        """初始化FinBERT模型，带重试机制"""
        
        model_name = 'ProsusAI/finbert'
        cache_dir = '../models/finbert_cache'
        os.makedirs(cache_dir, exist_ok=True)
        
        # 首先尝试本地缓存
        try:
            print(f"🔄 尝试从本地缓存加载FinBERT模型...")
            
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=True
            )
            
            self.finbert_model = AutoModel.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=True
            )
            
            print(f"✅ 从本地缓存加载FinBERT模型成功！")
            return
            
        except Exception as e:
            print(f"⚠️ 本地缓存加载失败: {e}")
        
        # 如果本地缓存失败，尝试联网下载
        for attempt in range(max_retries):
            try:
                print(f"🔄 尝试下载FinBERT模型 (第{attempt+1}次)...")
                
                # 设置更长的超时时间
                os.environ['HF_HUB_TIMEOUT'] = '300'  # 5分钟超时
                
                # 加载tokenizer
                self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    local_files_only=False,
                    trust_remote_code=True,
                    resume_download=True
                )
                
                # 加载模型
                self.finbert_model = AutoModel.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    local_files_only=False,
                    trust_remote_code=True,
                    resume_download=True
                )
                
                print(f"✅ FinBERT模型下载成功！")
                return
                
            except Exception as e:
                print(f"❌ 第{attempt+1}次下载失败: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ 等待10秒后重试...")
                    time.sleep(10)
                else:
                    print(f"⚠️ 多次尝试后仍无法加载FinBERT模型，将使用基础情绪分析")
                    self.finbert_tokenizer = None
                    self.finbert_model = None
        
    def load_news_data(self, market: str) -> pd.DataFrame:
        """加载指定市场的新闻数据"""
        
        news_dir = f'../dataset/{market}/news_sentiment'
        if not os.path.exists(news_dir):
            print(f"❌ 新闻数据目录不存在: {news_dir}")
            return pd.DataFrame()
        
        # 加载合并的日度情绪数据
        combined_file = f'{news_dir}/{market.lower()}_combined_daily_sentiment.csv'
        if os.path.exists(combined_file):
            df = pd.read_csv(combined_file)
            print(f"✅ 已加载 {market} 合并日度情绪数据: {len(df)} 条记录，{df['ticker'].nunique()} 只股票")
            return df
        
        # 如果没有合并文件，从详细数据中生成
        print(f"🔄 从详细新闻数据生成 {market} 的日度情绪数据...")
        return self._generate_daily_sentiment_from_detailed(news_dir, market)
    
    def _generate_daily_sentiment_from_detailed(self, news_dir: str, market: str) -> pd.DataFrame:
        """从详细新闻数据生成日度情绪数据 - 处理所有股票"""
        
        daily_data = []
        
        # 获取所有详细新闻文件
        news_files = [f for f in os.listdir(news_dir) if f.endswith('_news_detailed.csv')]
        print(f"🔍 发现 {len(news_files)} 个 {market} 新闻文件")
        
        for file in tqdm(news_files, desc=f"处理 {market} 新闻文件"):
            ticker = file.replace('_news_detailed.csv', '')
            file_path = os.path.join(news_dir, file)
            
            try:
                df = pd.read_csv(file_path)
                if len(df) == 0:
                    continue
                
                # 确保有发布日期列
                if 'published_utc' not in df.columns:
                    # 如果没有published_utc列，尝试其他可能的日期列
                    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                    if date_columns:
                        df['published_utc'] = df[date_columns[0]]
                    else:
                        print(f"⚠️ 文件 {file} 没有找到日期列，跳过")
                        continue
                
                # 转换日期格式
                df['date'] = pd.to_datetime(df['published_utc']).dt.date
                
                # 确保有情绪分析列
                if 'sentiment_polarity' not in df.columns:
                    # 如果没有情绪列，使用高级情绪分析
                    if 'title' in df.columns and 'description' in df.columns:
                        df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
                        
                        # 使用高级情绪分析
                        print(f"🔄 为 {ticker} 进行高级情绪分析...")
                        sentiment_results = []
                        for text in tqdm(df['text'], desc=f"分析 {ticker} 情绪", leave=False):
                            sentiment_features = self.extract_advanced_sentiment_features(str(text))
                            sentiment_results.append(sentiment_features)
                        
                        sentiment_df = pd.DataFrame(sentiment_results)
                        
                        # 使用组合情绪分数
                        if 'finbert_mean' in sentiment_df.columns:
                            # 如果有FinBERT特征，使用综合评分
                            df['sentiment_polarity'] = (
                                sentiment_df['textblob_polarity'] * 0.3 +
                                sentiment_df['vader_compound'] * 0.3 +
                                sentiment_df['finbert_mean'] * 0.4
                            )
                            df['sentiment_confidence'] = sentiment_df['emotion_confidence']
                        else:
                            # 否则使用TextBlob和VADER的组合
                            df['sentiment_polarity'] = (
                                sentiment_df['textblob_polarity'] * 0.5 +
                                sentiment_df['vader_compound'] * 0.5
                            )
                            df['sentiment_confidence'] = sentiment_df['emotion_confidence']
                        
                        df['urgency_score'] = df['text'].apply(lambda x: min(1.0, len(str(x)) / 100))
                        
                    else:
                        # 使用默认值
                        df['sentiment_polarity'] = 0.0
                        df['sentiment_confidence'] = 0.5
                        df['urgency_score'] = 0.5
                
                # 创建关键词特征
                text_columns = []
                if 'title' in df.columns:
                    text_columns.append('title')
                if 'description' in df.columns:
                    text_columns.append('description')
                if 'text' in df.columns:
                    text_columns.append('text')
                
                if text_columns:
                    df['combined_text'] = df[text_columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
                    
                    # 添加关键词特征
                    for category, keywords in self.news_type_keywords.items():
                        df[f'keywords_{category}'] = df['combined_text'].apply(
                            lambda x: sum(1 for keyword in keywords if keyword.lower() in str(x).lower())
                        )
                else:
                    # 使用默认关键词特征
                    for category in self.news_type_keywords.keys():
                        df[f'keywords_{category}'] = 0
                
                # 按日期分组聚合
                agg_dict = {
                    'sentiment_polarity': ['mean', 'std', 'count'],
                    'sentiment_confidence': 'mean',
                    'urgency_score': 'mean'
                }
                
                # 添加关键词聚合
                for category in self.news_type_keywords.keys():
                    if f'keywords_{category}' in df.columns:
                        agg_dict[f'keywords_{category}'] = 'sum'
                
                daily_agg = df.groupby('date').agg(agg_dict).reset_index()
                
                # 平铺列名
                daily_agg.columns = ['date'] + [f'{col[0]}_{col[1]}' if isinstance(col, tuple) and col[1] else col[0] 
                                              for col in daily_agg.columns[1:]]
                
                # 重命名列以保持一致性
                column_mapping = {}
                for col in daily_agg.columns:
                    if col.startswith('keywords_') and not col.endswith('_sum'):
                        column_mapping[col] = f'{col}_sum'
                
                if column_mapping:
                    daily_agg = daily_agg.rename(columns=column_mapping)
                
                # 添加股票代码
                daily_agg['ticker'] = ticker
                
                daily_data.append(daily_agg)
                
            except Exception as e:
                print(f"❌ 处理文件 {file} 时出错: {e}")
                continue
        
        if daily_data:
            combined_df = pd.concat(daily_data, ignore_index=True)
            
            # 保存合并的日度数据
            output_file = os.path.join(news_dir, f'{market.lower()}_combined_daily_sentiment.csv')
            combined_df.to_csv(output_file, index=False)
            print(f"✅ 已保存 {market} 合并日度情绪数据到: {output_file}")
            print(f"   数据包含: {len(combined_df)} 条记录，{combined_df['ticker'].nunique()} 只股票")
            
            return combined_df
        else:
            print(f"❌ 没有找到 {market} 的有效新闻数据")
            return pd.DataFrame()
    
    def extract_advanced_sentiment_features(self, text: str) -> Dict:
        """提取高级情绪特征"""
        
        features = {}
        
        # TextBlob情绪分析
        blob = TextBlob(text)
        features['textblob_polarity'] = blob.sentiment.polarity
        features['textblob_subjectivity'] = blob.sentiment.subjectivity
        
        # VADER情绪分析
        vader_scores = self.vader_analyzer.polarity_scores(text)
        features['vader_compound'] = vader_scores['compound']
        features['vader_positive'] = vader_scores['pos']
        features['vader_negative'] = vader_scores['neg']
        features['vader_neutral'] = vader_scores['neu']
        
        # FinBERT情绪分析
        finbert_features = self.extract_finbert_sentiment(text)
        features.update(finbert_features)
        
        # 情绪强度特征
        features['emotion_intensity'] = abs(features['textblob_polarity'])
        features['emotion_confidence'] = max(vader_scores['pos'], vader_scores['neg'])
        
        # 文本特征
        features['text_length'] = len(text)
        features['sentence_count'] = len(text.split('.'))
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # 金融关键词特征
        features.update(self._extract_financial_keywords(text))
        
        return features
    
    def _extract_financial_keywords(self, text: str) -> Dict:
        """提取金融相关关键词特征"""
        
        text_lower = text.lower()
        features = {}
        
        for category, keywords in self.news_type_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'keywords_{category}'] = count
        
        # 财务数字特征
        features['has_percentage'] = 1 if '%' in text else 0
        features['has_dollar_amount'] = 1 if '$' in text else 0
        features['number_count'] = len(re.findall(r'\d+', text))
        
        return features
    
    def generate_finbert_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """使用FinBERT生成文本嵌入"""
        
        if self.finbert_tokenizer is None or self.finbert_model is None:
            print("⚠️ FinBERT模型未加载，跳过嵌入生成")
            return np.array([])
        
        embeddings = []
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="生成FinBERT嵌入"):
                batch_texts = texts[i:i+batch_size]
                
                # 分词
                inputs = self.finbert_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # 生成嵌入
                with torch.no_grad():
                    outputs = self.finbert_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token
                
                embeddings.append(batch_embeddings)
            
            return np.vstack(embeddings) if embeddings else np.array([])
            
        except Exception as e:
            print(f"❌ FinBERT嵌入生成失败: {e}")
            return np.array([])
    
    def extract_finbert_sentiment(self, text: str) -> Dict:
        """使用FinBERT提取情绪特征"""
        
        if self.finbert_tokenizer is None or self.finbert_model is None:
            return {}
        
        try:
            # 分词
            inputs = self.finbert_tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )
            
            # 获取嵌入
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
            # 从嵌入中提取情绪特征（简化版）
            sentiment_features = {
                'finbert_positive': float(np.mean(embeddings[embeddings > 0])) if np.any(embeddings > 0) else 0.0,
                'finbert_negative': float(np.mean(embeddings[embeddings < 0])) if np.any(embeddings < 0) else 0.0,
                'finbert_magnitude': float(np.linalg.norm(embeddings)),
                'finbert_mean': float(np.mean(embeddings)),
                'finbert_std': float(np.std(embeddings))
            }
            
            return sentiment_features
            
        except Exception as e:
            print(f"❌ FinBERT情绪提取失败: {e}")
            return {}
    
    def create_time_series_features(self, df: pd.DataFrame, window_sizes: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """创建时间序列特征"""
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
        
        sentiment_cols = ['sentiment_polarity_mean', 'sentiment_confidence_mean', 'urgency_score_mean']
        
        for ticker in tqdm(df['ticker'].unique(), desc="生成时序特征"):
            ticker_mask = df['ticker'] == ticker
            ticker_data = df[ticker_mask].copy()
            
            for window in window_sizes:
                for col in sentiment_cols:
                    if col in ticker_data.columns:
                        # 滚动均值
                        df.loc[ticker_mask, f'{col}_ma_{window}'] = ticker_data[col].rolling(window=window, min_periods=1).mean()
                        
                        # 滚动标准差
                        df.loc[ticker_mask, f'{col}_std_{window}'] = ticker_data[col].rolling(window=window, min_periods=1).std()
                        
                        # 滚动趋势
                        df.loc[ticker_mask, f'{col}_trend_{window}'] = ticker_data[col].rolling(window=window, min_periods=2).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
                        )
        
        return df
    
    def align_with_price_data(self, sentiment_df: pd.DataFrame, price_data_path: str) -> pd.DataFrame:
        """将情绪数据与价格数据对齐"""
        
        # 加载价格数据
        if not os.path.exists(price_data_path):
            print(f"❌ 价格数据文件不存在: {price_data_path}")
            return sentiment_df
        
        price_df = pd.read_csv(price_data_path)
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        
        # 合并数据
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        merged_df = pd.merge(
            price_df,
            sentiment_df,
            left_on=['Date', 'Ticker'],
            right_on=['date', 'ticker'],
            how='left'
        )
        
        # 填充缺失的情绪数据
        sentiment_cols = [col for col in merged_df.columns if 'sentiment' in col.lower() or 'urgency' in col.lower()]
        
        for col in sentiment_cols:
            merged_df[col] = merged_df.groupby('Ticker')[col].fillna(method='ffill').fillna(0)
        
        return merged_df
    
    def create_market_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建市场上下文特征"""
        
        df = df.copy()
        
        # 市场整体情绪
        df['market_sentiment_mean'] = df.groupby('date')['sentiment_polarity_mean'].transform('mean')
        df['market_sentiment_std'] = df.groupby('date')['sentiment_polarity_mean'].transform('std')
        
        # 相对情绪（个股相对于市场）
        df['relative_sentiment'] = df['sentiment_polarity_mean'] - df['market_sentiment_mean']
        
        # 新闻活跃度
        df['market_news_volume'] = df.groupby('date')['sentiment_polarity_count'].transform('sum')
        df['relative_news_volume'] = df['sentiment_polarity_count'] / df['market_news_volume']
        
        # 情绪分歧度
        df['sentiment_divergence'] = abs(df['relative_sentiment'])
        
        return df
    
    def process_all_markets(self) -> Dict[str, pd.DataFrame]:
        """处理所有市场的新闻数据"""
        
        processed_data = {}
        
        for market in self.markets:
            print(f"\n🔄 开始处理 {market} 市场的新闻数据...")
            
            # 加载原始新闻数据
            news_df = self.load_news_data(market)
            
            if news_df.empty:
                print(f"❌ {market} 市场没有可用的新闻数据")
                continue
            
            # 创建时间序列特征
            print(f"🔄 为 {market} 创建时间序列特征...")
            news_df = self.create_time_series_features(news_df)
            
            # 创建市场上下文特征
            print(f"🔄 为 {market} 创建市场上下文特征...")
            news_df = self.create_market_context_features(news_df)
            
            # 与价格数据对齐
            price_data_path = f'../dataset/{market}/{market.lower()}_price_data.csv'
            if os.path.exists(price_data_path):
                print(f"🔄 将 {market} 的新闻数据与价格数据对齐...")
                aligned_df = self.align_with_price_data(news_df, price_data_path)
                processed_data[market] = aligned_df
            else:
                print(f"⚠️ 未找到 {market} 的价格数据，使用纯新闻数据")
                processed_data[market] = news_df
            
            print(f"✅ {market} 数据处理完成: {len(processed_data[market])} 条记录")
        
        return processed_data
    
    def save_processed_data(self, processed_data: Dict[str, pd.DataFrame], output_dir: str = '../dataset/processed_sentiment'):
        """保存处理后的数据"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        for market, df in processed_data.items():
            output_file = os.path.join(output_dir, f'{market.lower()}_processed_sentiment.csv')
            df.to_csv(output_file, index=False)
            print(f"✅ 已保存 {market} 处理后的数据到: {output_file}")
        
        # 保存数据统计信息
        stats = {}
        for market, df in processed_data.items():
            stats[market] = {
                'total_records': len(df),
                'date_range': f"{df['date'].min()} 到 {df['date'].max()}",
                'unique_tickers': df['ticker'].nunique() if 'ticker' in df.columns else 0,
                'feature_count': len(df.columns)
            }
        
        stats_file = os.path.join(output_dir, 'processing_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 已保存处理统计信息到: {stats_file}")

def main():
    """主函数"""
    
    # 简化配置，移除FinBERT相关
    config = {
        'time_windows': [3, 7, 14, 30],
        'batch_size': 32,
        'process_all_stocks': True  # 确保处理所有股票
    }
    
    # 创建预处理器
    print("🚀 开始处理所有市场的新闻情绪数据...")
    preprocessor = NewsDataPreprocessor(config)
    
    # 处理所有市场数据
    processed_data = preprocessor.process_all_markets()
    
    # 保存处理后的数据
    preprocessor.save_processed_data(processed_data)
    
    print("\n🎉 新闻情绪数据预处理完成！")
    print("\n📊 数据统计:")
    for market, df in processed_data.items():
        if not df.empty:
            print(f"  {market}: {len(df)} 条记录, {df['ticker'].nunique()} 只股票, {len(df.columns)} 个特征")
        else:
            print(f"  {market}: 无数据")

if __name__ == "__main__":
    main() 