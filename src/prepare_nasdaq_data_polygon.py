import requests
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import pickle
import re

# Polygon.io API配置 - 升级版：无限制调用 + 5年历史数据
POLYGON_API_KEY = "NoJWwFDrkW3bh0oyp1Wc8ZhYjLu_0zZX"  # 用户的升级版API密钥
POLYGON_BASE_URL = "https://api.polygon.io"

def filter_high_quality_stocks(all_stocks, target_count=1000):
    """筛选高质量的NASDAQ股票，控制在目标数量左右"""
    print(f"正在筛选高质量股票，目标数量: {target_count}...")
    
    # 定义知名的大市值NASDAQ股票（优先级最高）
    priority_tickers = {
        # 大型科技股
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE',
        'CRM', 'ORCL', 'INTC', 'CSCO', 'AVGO', 'TXN', 'QCOM', 'AMD', 'MU', 'AMAT',
        'LRCX', 'ADI', 'KLAC', 'MRVL', 'NXPI', 'SNPS', 'CDNS', 'MCHP', 'FTNT', 'PANW',
        
        # 大型消费股
        'COST', 'SBUX', 'MDLZ', 'KHC', 'MNST', 'PEP', 'WBA', 'DLTR', 'ROST', 'FAST',
        'PAYX', 'VRSK', 'CTAS', 'ODFL', 'CPRT', 'ORLY', 'LULU', 'MAR', 'ABNB',
        
        # 大型生物技术股
        'AMGN', 'BIIB', 'GILD', 'REGN', 'VRTX', 'BMRN', 'ALXN', 'CELG', 'ILMN', 'INCY',
        'ISRG', 'ALGN', 'IDXX', 'DXCM', 'ZBH', 'TECH', 'MRNA', 'BNTX',
        
        # 大型金融科技股
        'PYPL', 'ADSK', 'FISV', 'ADP', 'INTU', 'PAYX', 'CTSH', 'AKAM', 'VRSN',
        
        # 大型媒体和通信股
        'CMCSA', 'CHTR', 'TMUS', 'DISH', 'FOXA', 'FOX', 'LBRDA', 'LBRDK', 'SIRI',
        
        # 大型工业和公用事业股
        'HON', 'ITW', 'EMR', 'XEL', 'EXC', 'AEP', 'ATO', 'CMS', 'DTE', 'ES',
        
        # 知名中概股
        'BABA', 'JD', 'BIDU', 'NTES', 'BILI', 'IQ', 'PDD', 'NIO', 'XPEV', 'LI',
        
        # 新兴科技股
        'ZM', 'DOCU', 'OKTA', 'SPLK', 'CRM', 'NOW', 'WDAY', 'VEEV', 'ZS', 'CRWD',
        'DDOG', 'NET', 'FSLY', 'MDB', 'ESTC', 'CFLT', 'SNOW', 'PLTR', 'RBLX', 'COIN',
        
        # 电商和共享经济
        'EBAY', 'ETSY', 'SHOP', 'SE', 'UBER', 'LYFT', 'DASH', 'GRUB',
        
        # 生物技术和医疗设备
        'TMO', 'DHR', 'A', 'BDX', 'BSX', 'SYK', 'EW', 'HOLX', 'VAR', 'PKI',
        
        # 半导体设备和材料
        'ASML', 'WDC', 'STX', 'SWKS', 'QRVO', 'CRUS', 'MKSI', 'COHU', 'FORM', 'ENTG'
    }
    
    # 1. 首先选择所有优先级股票
    priority_stocks = []
    regular_stocks = []
    
    for stock in all_stocks:
        ticker = stock['ticker']
        if ticker in priority_tickers:
            priority_stocks.append(stock)
        else:
            regular_stocks.append(stock)
    
    print(f"找到 {len(priority_stocks)} 只优先级股票")
    print(f"剩余 {len(regular_stocks)} 只普通股票")
    
    # 2. 对普通股票进行筛选，优先选择看起来更有质量的
    # 筛选条件：
    # - 排除明显的垃圾股票（如权证、单位等）
    # - 优先选择较短的股票代码（通常是更成熟的公司）
    quality_regular_stocks = []
    
    for stock in regular_stocks:
        ticker = stock['ticker']
        name = stock['name'].lower()
        
        # 排除条件
        skip_reasons = []
        
        # 排除权证、单位等
        if any(word in name for word in ['warrant', 'unit', 'right', 'acquisition corp', 'acquisition company']):
            skip_reasons.append('special_instrument')
        
        # 排除过长的股票代码（通常是特殊工具）
        if len(ticker) > 5:
            skip_reasons.append('long_ticker')
            
        # 排除包含特殊字符的股票代码
        if not ticker.isalpha():
            skip_reasons.append('special_chars')
            
        # 如果没有排除原因，则加入候选列表
        if not skip_reasons:
            quality_regular_stocks.append(stock)
    
    print(f"筛选出 {len(quality_regular_stocks)} 只高质量普通股票")
    
    # 3. 按字母顺序排序，确保一致性
    priority_stocks.sort(key=lambda x: x['ticker'])
    quality_regular_stocks.sort(key=lambda x: x['ticker'])
    
    # 4. 组合结果，确保总数接近目标
    final_stocks = priority_stocks.copy()
    
    # 计算还需要多少只股票
    remaining_needed = target_count - len(priority_stocks)
    
    if remaining_needed > 0 and quality_regular_stocks:
        # 从高质量普通股票中选择
        additional_stocks = quality_regular_stocks[:remaining_needed]
        final_stocks.extend(additional_stocks)
        print(f"添加了 {len(additional_stocks)} 只普通股票")
    
    # 最终按字母顺序排序
    final_stocks.sort(key=lambda x: x['ticker'])
    
    print(f"最终筛选结果: {len(final_stocks)} 只股票")
    print(f"其中优先级股票: {len(priority_stocks)} 只")
    print(f"其中普通股票: {len(final_stocks) - len(priority_stocks)} 只")
    
    return final_stocks
 
def get_all_nasdaq_tickers_from_api():
    """直接从Polygon.io API获取所有NASDAQ交易所上市的股票"""
    print("正在从Polygon.io API获取所有NASDAQ股票列表...")
    
    all_nasdaq_tickers = []
    
    # 使用Polygon.io的tickers API获取所有NASDAQ股票
    url = f"{POLYGON_BASE_URL}/v3/reference/tickers"
    
    # 参数设置：只获取stocks类型，只获取NASDAQ交易所，只获取活跃股票
    params = {
        'market': 'stocks',
        'active': 'true',
        'sort': 'ticker',
        'order': 'asc',
        'limit': 1000,  # 单次最大返回1000条
        'apikey': POLYGON_API_KEY
    }
    
    page_count = 0
    
    try:
        while True:
            page_count += 1
            print(f"正在获取第 {page_count} 页数据...")
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                print("API速率限制，等待60秒...")
                time.sleep(60)
                response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"API请求失败: {response.status_code}")
                break
            
            data = response.json()
            
            if 'results' not in data or not data['results']:
                print("没有更多数据")
                break
            
            # 过滤出NASDAQ交易所的股票
            nasdaq_stocks = []
            for stock in data['results']:
                # 检查主要交易所是否为NASDAQ相关
                primary_exchange = stock.get('primary_exchange', '').upper()
                ticker = stock.get('ticker', '')
                name = stock.get('name', '')
                
                # NASDAQ相关的交易所代码
                nasdaq_exchanges = ['XNAS', 'XNGS', 'XNMS', 'XNDQ']
                
                if primary_exchange in nasdaq_exchanges:
                    nasdaq_stocks.append({
                        'ticker': ticker,
                        'name': name,
                        'primary_exchange': primary_exchange,
                        'market': stock.get('market', ''),
                        'locale': stock.get('locale', ''),
                        'currency_name': stock.get('currency_name', ''),
                        'active': stock.get('active', True)
                    })
            
            all_nasdaq_tickers.extend(nasdaq_stocks)
            print(f"第 {page_count} 页: 找到 {len(nasdaq_stocks)} 只NASDAQ股票")
            
            # 检查是否有下一页
            if 'next_url' not in data or not data['next_url']:
                break
            
            # 解析next_url的参数来继续获取下一页
            next_url = data['next_url']
            if 'cursor=' in next_url:
                cursor = next_url.split('cursor=')[1].split('&')[0]
                params['cursor'] = cursor
            else:
                break
            
            # 避免过快请求
            time.sleep(2)
    
    except Exception as e:
        print(f"获取NASDAQ股票列表时出错: {e}")
    
    # 去重并排序
    unique_tickers = {}
    for stock in all_nasdaq_tickers:
        ticker = stock['ticker']
        if ticker not in unique_tickers:
            unique_tickers[ticker] = stock
    
    final_nasdaq_list = list(unique_tickers.values())
    final_nasdaq_list.sort(key=lambda x: x['ticker'])
    
    print(f"\n原始获得 {len(final_nasdaq_list)} 只真实NASDAQ股票")
    
    # 筛选高质量股票，控制在1000只左右
    filtered_nasdaq_list = filter_high_quality_stocks(final_nasdaq_list)
    
    print(f"\n筛选后获得 {len(filtered_nasdaq_list)} 只高质量NASDAQ股票")
    
    # 显示前10只和后10只
    if filtered_nasdaq_list:
        print("前10只股票:")
        for i in range(min(10, len(filtered_nasdaq_list))):
            stock = filtered_nasdaq_list[i]
            print(f"  {stock['ticker']}: {stock['name']} ({stock['primary_exchange']})")
        
        print("最后10只股票:")
        for i in range(max(0, len(filtered_nasdaq_list)-10), len(filtered_nasdaq_list)):
            stock = filtered_nasdaq_list[i]
            print(f"  {stock['ticker']}: {stock['name']} ({stock['primary_exchange']})")
    
    # 只返回股票代码列表
    ticker_list = [stock['ticker'] for stock in filtered_nasdaq_list]
    
    # 创建表格用于兼容性
    nasdaq_table = pd.DataFrame({
        'Ticker': ticker_list,
        'Security': [stock['name'] for stock in filtered_nasdaq_list],
        'GICS Sector': [''] * len(filtered_nasdaq_list)
    })
    
    return ticker_list, nasdaq_table

def fetch_stock_data_polygon(ticker, start_date, end_date):
    """使用Polygon.io API获取单只股票的OHLCVA数据 - 升级版无限制API"""
    try:
        # 构建API URL
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,  # 支持更大的数据量
            'apikey': POLYGON_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        # 升级版API不再有严格的速率限制，但仍保留基本的重试机制
        if response.status_code == 429:
            print(f"偶尔的速率限制，等待5秒... ({ticker})")
            time.sleep(5)  # 大幅减少等待时间
            response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"API请求失败: {response.status_code} - {ticker}")
            return None
            
        data = response.json()
        
        if 'results' not in data or not data['results']:
            print(f"无数据返回: {ticker}")
            return None
        
        # 解析数据
        results = data['results']
        df_data = []
        
        for bar in results:
            # polygon.io返回的时间戳是毫秒，需要转换
            date = pd.to_datetime(bar['t'], unit='ms')
            df_data.append({
                'Date': date,
                'Open': bar['o'],
                'High': bar['h'], 
                'Low': bar['l'],
                'Volume': bar['v'],
                'Adj Close': bar['c']  # polygon.io的close已经是复权价格
            })
        
        if not df_data:
            return None
            
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df = df.sort_index()  # 确保按日期排序
        
        print(f"✓ {ticker}: {len(df)} 天数据 (从 {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')})")
        
        return df[['Open', 'High', 'Low', 'Volume', 'Adj Close']]
        
    except Exception as e:
        print(f"获取数据失败: {ticker}, 错误: {e}")
        return None

def download_batch(batch_tickers, batch_num, start_date, end_date, cache_dir, csv_dir):
    """下载一批股票数据 - 升级版API无限制调用"""
    print(f"\n{'='*50}")
    print(f"开始下载第 {batch_num} 批数据 ({len(batch_tickers)} 只股票)")
    print(f"{'='*50}")
    
    batch_data = []
    successful_tickers = []
    
    for i, ticker in enumerate(tqdm(batch_tickers, desc=f"批次 {batch_num}")):
        stock_df = fetch_stock_data_polygon(ticker, start_date, end_date)
        
        # 数据质量检查
        if stock_df is not None and len(stock_df) > 0:
            valid_ratio = stock_df['Adj Close'].count() / len(stock_df)
            if valid_ratio > 0.5:  # 至少50%的数据有效
                batch_data.append(stock_df)
                successful_tickers.append(ticker)
                
                # 保存单个股票的CSV文件
                csv_filename = f"{ticker}.csv"
                csv_filepath = os.path.join(csv_dir, csv_filename)
                
                # 创建包含日期的完整DataFrame用于CSV导出
                csv_df = stock_df.copy()
                csv_df.index.name = 'Date'  # 确保索引有名称
                
                # 添加一些有用的计算列
                csv_df['Close'] = csv_df['Adj Close']  # 添加Close列以保持兼容性
                csv_df['Daily_Return'] = csv_df['Adj Close'].pct_change()  # 日收益率
                csv_df['Ticker'] = ticker  # 添加股票代码列
                
                # 重新排序列
                column_order = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return']
                csv_df = csv_df[column_order]
                
                # 保存CSV文件
                csv_df.to_csv(csv_filepath, index=True)
                
                print(f"✓ {ticker} (有效率: {valid_ratio:.1%}, {len(stock_df)}天) - CSV已保存")
            else:
                print(f"✗ {ticker} (有效率过低: {valid_ratio:.1%})")
        else:
            print(f"✗ {ticker} (下载失败或无数据)")
        
        # 升级版API无限制，只需要很短的间隔避免过于密集的请求
        if i < len(batch_tickers) - 1:  # 最后一个不需要等待
            time.sleep(0.1)  # 仅100毫秒间隔，避免请求过于密集
    
    # 保存批次数据到缓存
    if batch_data:
        batch_cache_file = os.path.join(cache_dir, f'batch_{batch_num}_data.pkl')
        tickers_cache_file = os.path.join(cache_dir, f'batch_{batch_num}_tickers.pkl')
        
        with open(batch_cache_file, 'wb') as f:
            pickle.dump(batch_data, f)
        with open(tickers_cache_file, 'wb') as f:
            pickle.dump(successful_tickers, f)
            
        print(f"批次 {batch_num} 完成: 成功下载 {len(successful_tickers)} 只股票")
        print(f"CSV文件已保存到: {csv_dir}")
    else:
        print(f"批次 {batch_num} 失败: 没有成功下载任何股票")
    
    return len(successful_tickers)

def align_all_data(all_dataframes):
    """对齐所有股票的数据到统一的时间索引"""
    if not all_dataframes:
        return [], None
    
    # 找到所有数据的公共时间范围
    all_dates = set()
    for df in all_dataframes:
        all_dates.update(df.index)
    
    # 创建统一的时间索引
    unified_index = pd.DatetimeIndex(sorted(all_dates))
    print(f"统一时间索引: {len(unified_index)} 天 (从 {unified_index.min().strftime('%Y-%m-%d')} 到 {unified_index.max().strftime('%Y-%m-%d')})")
    
    # 对齐所有数据
    aligned_data = []
    for df in all_dataframes:
        aligned_df = df.reindex(unified_index)
        # 填充NaN值为前一个有效值
        aligned_df = aligned_df.fillna(method='ffill').fillna(method='bfill')
        aligned_data.append(aligned_df.to_numpy(dtype=np.float32))
    
    return aligned_data, unified_index

def load_all_batches(cache_dir, num_batches):
    """加载所有批次的数据"""
    print(f"\n加载所有 {num_batches} 个批次的数据...")
    
    all_stocks_data_list = []
    all_successful_tickers = []
    
    for batch_num in range(1, num_batches + 1):
        batch_cache_file = os.path.join(cache_dir, f'batch_{batch_num}_data.pkl')
        tickers_cache_file = os.path.join(cache_dir, f'batch_{batch_num}_tickers.pkl')
        
        if os.path.exists(batch_cache_file) and os.path.exists(tickers_cache_file):
            with open(batch_cache_file, 'rb') as f:
                batch_data = pickle.load(f)
            with open(tickers_cache_file, 'rb') as f:
                batch_tickers = pickle.load(f)
            
            all_stocks_data_list.extend(batch_data)
            all_successful_tickers.extend(batch_tickers)
            print(f"加载批次 {batch_num}: {len(batch_tickers)} 只股票")
        else:
            print(f"警告: 批次 {batch_num} 数据不存在")
    
    return all_stocks_data_list, all_successful_tickers

def main():
    """主执行函数"""
    # --- 配置信息 ---
    OUTPUT_BASE_DIR = '../dataset'
    MARKET_NAME = 'NASDAQ'
    OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, MARKET_NAME)
    CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
    CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    
    # 设置时间范围 - 升级版API支持5年历史数据
    START_DATE_STR = '2020-01-01'  # 扩展到5年数据
    END_DATE_STR = '2024-12-31'
    
    # 批次设置 - 升级版API无限制，可以使用更大的批次
    BATCH_SIZE = 200  # 增大批次大小，加快下载速度
    
    print(f"数据将保存到: {os.path.abspath(OUTPUT_DIR)}")
    print(f"数据时间范围: {START_DATE_STR} to {END_DATE_STR} (5年历史数据)")
    print(f"✅ 升级版API：无限制调用 + 5年历史数据支持")

    # 1. 从Polygon.io API获取所有NASDAQ股票列表
    nasdaq_tickers, nasdaq_table = get_all_nasdaq_tickers_from_api()
    
    if not nasdaq_tickers:
        print("错误：未能从API获取到NASDAQ股票列表")
        return
    
    actual_stock_count = len(nasdaq_tickers)
    print(f"\n实际将下载 {actual_stock_count} 只真实NASDAQ股票")
    print(f"批次设置: 每批 {BATCH_SIZE} 只股票，共 {(actual_stock_count + BATCH_SIZE - 1) // BATCH_SIZE} 批")
    print(f"预计总时间: 约 {((actual_stock_count + BATCH_SIZE - 1) // BATCH_SIZE) * 2} 分钟 (升级版API，大幅提速)")

    # 2. 分批下载数据
    num_batches = (len(nasdaq_tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"将分 {num_batches} 批下载数据")
    
    total_downloaded = 0
    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(nasdaq_tickers))
        batch_tickers = nasdaq_tickers[start_idx:end_idx]
        
        # 检查是否已经下载过
        batch_cache_file = os.path.join(CACHE_DIR, f'batch_{batch_num}_data.pkl')
        if os.path.exists(batch_cache_file):
            print(f"批次 {batch_num} 已存在，跳过下载")
            continue
        
        downloaded_count = download_batch(
            batch_tickers, batch_num, START_DATE_STR, END_DATE_STR, 
            CACHE_DIR, CSV_DIR
        )
        total_downloaded += downloaded_count
        
        # 批次间稍作等待，避免过于密集的请求
        if batch_num < num_batches:
            print(f"\n等待 5 秒后下载下一批...")  # 大幅减少等待时间
            time.sleep(5)

    # 3. 加载所有批次数据
    all_stocks_dataframes, all_successful_tickers = load_all_batches(CACHE_DIR, num_batches)
    
    if not all_stocks_dataframes:
        print("错误：未能成功获取任何股票的数据。")
        return

    print(f"\n总计成功获取 {len(all_stocks_dataframes)} 只股票的数据")

    # 4. 对齐所有股票数据
    aligned_data, unified_index = align_all_data(all_stocks_dataframes)
    
    if not aligned_data:
        print("数据对齐失败")
        return

    # 5. 生成 nasdaq_ticker.csv
    ticker_df = pd.DataFrame({
        'Symbol': all_successful_tickers,
        'Name': nasdaq_table.get('Security', [''] * len(all_successful_tickers))[:len(all_successful_tickers)],
        'Sector': nasdaq_table.get('GICS Sector', [''] * len(all_successful_tickers))[:len(all_successful_tickers)]
    })
    
    ticker_csv_path = os.path.join(OUTPUT_DIR, 'nasdaq_ticker.csv')
    ticker_df.to_csv(ticker_csv_path, index=False)
    print(f"文件已保存: {ticker_csv_path}")

    # 6. 生成 NASDAQ.npy (与SP500.npy格式一致)
    nasdaq_data_np = np.stack(aligned_data, axis=0)
    nasdaq_data_np[np.isnan(nasdaq_data_np)] = 1.1  # 填充NaN值，与SP500一致
    
    nasdaq_npy_path = os.path.join(OUTPUT_DIR, 'NASDAQ.npy')
    np.save(nasdaq_npy_path, nasdaq_data_np)
    print(f"文件已保存: {nasdaq_npy_path}, 形状: {nasdaq_data_np.shape}")

    # 7. 创建简化的基准数据 (与SP500一致)
    baseline_data_np = np.mean(nasdaq_data_np, axis=0)  # 使用所有股票的平均值作为基准
    baseline_npy_path = os.path.join(OUTPUT_DIR, 'baseline_data_nasdaq.npy')
    np.save(baseline_npy_path, baseline_data_np)
    print(f"文件已保存: {baseline_npy_path}, 形状: {baseline_data_np.shape}")

    print("\n所有文件生成完毕！")
    print(f"生成文件列表:")
    print(f"  - nasdaq_ticker.csv: {len(all_successful_tickers)} 只股票")
    print(f"  - NASDAQ.npy: 形状 {nasdaq_data_np.shape}")
    print(f"  - baseline_data_nasdaq.npy: 形状 {baseline_data_np.shape}")
    print(f"  - CSV文件目录: {CSV_DIR} (包含 {len(all_successful_tickers)} 个股票CSV文件)")
    
    # 验证数据质量
    print(f"\n数据质量报告 (升级版5年数据):")
    print(f"  - 股票数量: {len(all_successful_tickers)} (全部为Polygon.io API提供的真实NASDAQ股票)")
    print(f"  - 时间范围: {unified_index.min().strftime('%Y-%m-%d')} 到 {unified_index.max().strftime('%Y-%m-%d')}")
    print(f"  - 交易日数: {len(unified_index)}")
    print(f"  - 特征数: {nasdaq_data_np.shape[2]}")
    print(f"  - 数据有效率: {np.count_nonzero(~np.isnan(nasdaq_data_np.sum(axis=2))) / (nasdaq_data_np.shape[0] * nasdaq_data_np.shape[1]):.2%}")
    print(f"✅ 使用升级版API：5年历史数据 (2020-2024) + 无限制调用")
    print(f"✅ 数据格式与SP500一致，可以使用相同的训练逻辑")
    
    # 8. 检查CSV文件示例
    if len(all_successful_tickers) > 0:
        sample_ticker = all_successful_tickers[0]
        sample_csv_path = os.path.join(CSV_DIR, f'{sample_ticker}.csv')
        if os.path.exists(sample_csv_path):
            print(f"\n示例CSV文件 ({sample_ticker}.csv) 前5行:")
            sample_df = pd.read_csv(sample_csv_path, index_col=0)
            print(sample_df.head().to_string())

if __name__ == '__main__':
    main() 