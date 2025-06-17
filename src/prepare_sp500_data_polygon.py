import requests
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import exchange_calendars as xcals
import pickle

# Polygon.io API配置 - 升级版：无限制调用 + 5年历史数据
POLYGON_API_KEY = "NoJWwFDrkW3bh0oyp1Wc8ZhYjLu_0zZX"  # 用户的升级版API密钥
POLYGON_BASE_URL = "https://api.polygon.io"

def get_sp500_tickers():
    """获取S&P 500成分股列表"""
    print("正在获取S&P 500成分股列表...")
    try:
        # 从维基百科获取SP500成分股列表
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        
        # 清理股票代码
        tickers = sp500_table['Symbol'].tolist()
        # 替换一些特殊字符 (polygon.io使用标准格式)
        tickers = [ticker.replace('.', '.') for ticker in tickers]  # 保持原样
        
        print(f"成功获取 {len(tickers)} 只S&P 500成分股")
        return tickers, sp500_table
    except Exception as e:
        print(f"从维基百科获取失败: {e}")
        # 备用的SP500主要成分股列表
        print("使用备用的S&P 500主要成分股列表...")
        backup_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'TSLA', 'META', 'BRK.B', 'UNH',
            'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'CVX', 'HD', 'ABBV', 'PFE',
            'KO', 'AVGO', 'PEP', 'COST', 'WMT', 'TMO', 'BAC', 'NFLX', 'CRM', 'ABT',
            'ADBE', 'MRK', 'ACN', 'VZ', 'WFC', 'CSCO', 'DHR', 'TXN', 'NKE', 'QCOM',
            'BMY', 'PM', 'RTX', 'NEE', 'AMGN', 'COP', 'UNP', 'T', 'LOW', 'HON',
            'SPGI', 'INTU', 'UPS', 'AMD', 'CAT', 'GS', 'BA', 'ELV', 'PGR', 'AMAT',
            'DE', 'BKNG', 'GILD', 'AXP', 'BLK', 'MDLZ', 'TJX', 'SYK', 'ADP', 'VRTX',
            'ADI', 'CVS', 'SCHW', 'TMUS', 'AMT', 'ZTS', 'EOG', 'LMT', 'PYPL', 'REGN',
            'C', 'MU', 'NOW', 'GE', 'MMM', 'CB', 'SO', 'ISRG', 'DUK', 'BSX',
            'KLAC', 'PLD', 'ICE', 'FI', 'AON', 'CCI', 'LRCX', 'APD', 'SHW', 'CME',
            'MSCI', 'MCO', 'ITW', 'FIS', 'COF', 'WM', 'CL', 'GD', 'USB', 'TFC',
            'EMR', 'NSC', 'MMC', 'PSA', 'FDX', 'AEP', 'SRE', 'D', 'EXC', 'XEL',
            'WEC', 'ES', 'AEE', 'LNT', 'NI', 'EVRG', 'CNP', 'ETR', 'FE', 'AES',
            'PPL', 'PEG', 'EIX', 'ED', 'DTE', 'PCG', 'PNW', 'CMS', 'AWK', 'ATO',
            'WTR', 'SJM', 'K', 'CPB', 'GIS', 'HSY', 'HRL', 'CAG', 'MKC', 'CLX',
            'KMB', 'UL', 'EL', 'DPS', 'STZ', 'BF.B', 'TAP', 'SAM', 'MNST', 'KDP',
            'CCEP', 'WBA', 'CI', 'ANTM', 'HUM', 'CNC', 'MOH', 'EW', 'BDX', 'BAX',
            'ZBH', 'STE', 'HOLX', 'TFX', 'COO', 'ALGN', 'IQV', 'A', 'LH', 'DVN',
            'HAL', 'SLB', 'BKR', 'OXY', 'KMI', 'WMB', 'OKE', 'LNG', 'FANG', 'MRO',
            'APA', 'CTRA', 'HES', 'VLO', 'MPC', 'PSX', 'PXD', 'NXPI', 'MRVL', 'FTNT',
            'PANW', 'SNPS', 'CDNS', 'ANSS', 'KEYS', 'TYL', 'EPAM', 'CTSH', 'GLW', 'HPQ',
            'NTAP', 'STX', 'WDC', 'AKAM', 'JNPR', 'FFIV', 'NLOK', 'GEN', 'ZBRA', 'MPWR',
            'TER', 'SWKS', 'QRVO', 'IPG', 'OMC', 'NWSA', 'NWSA', 'FOXA', 'FOX', 'DIS',
            'CMCSA', 'CHTR', 'EA', 'ATVI', 'TTWO', 'NFLX', 'ROKU', 'MTCH', 'PINS', 'SNAP',
            'TWTR', 'SPOT', 'UBER', 'LYFT', 'DASH', 'ABNB', 'AIRB', 'BKNG', 'EXPE', 'TXG',
            'MAR', 'HLT', 'MGM', 'LVS', 'WYNN', 'CCL', 'RCL', 'NCLH', 'AAL', 'DAL',
            'UAL', 'LUV', 'ALK', 'JBLU', 'SAVE', 'UPS', 'FDX', 'XPO', 'ODFL', 'CHRW',
            'KNX', 'EXPD', 'JBHT', 'SAIA', 'LSTR', 'ARCB', 'WERN', 'KSU', 'UNP', 'CSX',
            'NSC', 'CP', 'CNI', 'RAIL', 'GWR', 'GATX', 'TRN', 'WAB', 'LECO', 'HON',
            'MMM', 'CAT', 'DE', 'CMI', 'EMR', 'ETN', 'GE', 'ITW', 'JCI', 'MAS',
            'ROK', 'SNA', 'TXT', 'WHR', 'XYL', 'AME', 'CTAS', 'FLS', 'FTV', 'GGG',
            'GNRC', 'HII', 'IR', 'IEX', 'LDOS', 'MHK', 'NDSN', 'OTIS', 'PCAR', 'PH',
            'PNR', 'PWR', 'RSG', 'RTX', 'TDG', 'TT', 'URI', 'VMI', 'WM', 'WSO',
            'ALB', 'APD', 'BALL', 'BLL', 'CF', 'DOW', 'DD', 'ECL', 'EMN', 'FCX',
            'FMC', 'IFF', 'IP', 'LIN', 'LYB', 'MLM', 'MOS', 'NEM', 'NUE', 'PKG',
            'PPG', 'SEE', 'SHW', 'VMC', 'WRK', 'AA', 'STLD', 'RS', 'CLF', 'X',
            'AOS', 'AYI', 'BWA', 'CARR', 'CHRW', 'CPRT', 'CSL', 'CUMMINS', 'DOV', 'EMR',
            'FAST', 'FBHS', 'FLS', 'GNRC', 'GWW', 'HON', 'HWM', 'IEX', 'ITW', 'JCI',
            'LECO', 'MAS', 'MLM', 'MMM', 'MSA', 'NLSN', 'OTIS', 'PAYC', 'PCAR', 'PH',
            'PNR', 'PWR', 'ROK', 'RSG', 'RTX', 'SNPS', 'SNA', 'SWK', 'TT', 'TXN',
            'URI', 'VMC', 'WM', 'WSO', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS'
        ]
        
        # 去重并限制数量
        backup_tickers = list(dict.fromkeys(backup_tickers))  # 去重
        if len(backup_tickers) < 500:
            print(f"警告：备用列表只有 {len(backup_tickers)} 只股票，少于请求的500只")
        
        # 创建简化的表格
        backup_table = pd.DataFrame({
            'Symbol': backup_tickers,
            'Security': [''] * len(backup_tickers),
            'GICS Sector': [''] * len(backup_tickers)
        })
        
        return backup_tickers, backup_table

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
    MARKET_NAME = 'SP500'
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
    BATCH_SIZE = 50  # 增大批次大小，加快下载速度
    MAX_STOCKS = 500  # 下载500只股票
    
    print(f"数据将保存到: {os.path.abspath(OUTPUT_DIR)}")
    print(f"数据时间范围: {START_DATE_STR} to {END_DATE_STR} (5年历史数据)")
    print(f"✅ 升级版API：无限制调用 + 5年历史数据支持")
    print(f"批次设置: 每批 {BATCH_SIZE} 只股票，总计 {MAX_STOCKS} 只")
    print(f"预计总时间: 约 {(MAX_STOCKS // BATCH_SIZE) * 1} 分钟 (升级版API，大幅提速)")

    # 1. 获取S&P 500成分股列表
    sp500_tickers, sp500_table = get_sp500_tickers()
    
    # 限制股票数量
    if len(sp500_tickers) > MAX_STOCKS:
        sp500_tickers = sp500_tickers[:MAX_STOCKS]
        print(f"限制到前 {MAX_STOCKS} 只股票")

    # 2. 分批下载数据
    num_batches = (len(sp500_tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"将分 {num_batches} 批下载数据")
    
    total_downloaded = 0
    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(sp500_tickers))
        batch_tickers = sp500_tickers[start_idx:end_idx]
        
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
            print(f"\n等待 2 秒后下载下一批...")  # 大幅减少等待时间
            time.sleep(2)

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

    # 5. 生成 sp500_ticker.csv
    ticker_df = pd.DataFrame({
        'Symbol': all_successful_tickers,
        'Name': sp500_table.get('Security', [''] * len(all_successful_tickers))[:len(all_successful_tickers)],
        'Sector': sp500_table.get('GICS Sector', [''] * len(all_successful_tickers))[:len(all_successful_tickers)]
    })
    
    ticker_csv_path = os.path.join(OUTPUT_DIR, 'sp500_ticker.csv')
    ticker_df.to_csv(ticker_csv_path, index=False)
    print(f"文件已保存: {ticker_csv_path}")

    # 6. 生成 SP500.npy
    sp500_data_np = np.stack(aligned_data, axis=0)
    sp500_data_np[np.isnan(sp500_data_np)] = 1.1
    
    sp500_npy_path = os.path.join(OUTPUT_DIR, 'SP500.npy')
    np.save(sp500_npy_path, sp500_data_np)
    print(f"文件已保存: {sp500_npy_path}, 形状: {sp500_data_np.shape}")

    # 7. 创建简化的基准数据
    baseline_data_np = np.mean(sp500_data_np, axis=0)  # 使用所有股票的平均值作为基准
    baseline_npy_path = os.path.join(OUTPUT_DIR, 'baseline_data_sp500.npy')
    np.save(baseline_npy_path, baseline_data_np)
    print(f"文件已保存: {baseline_npy_path}, 形状: {baseline_data_np.shape}")

    print("\n所有文件生成完毕！")
    print(f"生成文件列表:")
    print(f"  - sp500_ticker.csv: {len(all_successful_tickers)} 只股票")
    print(f"  - SP500.npy: 形状 {sp500_data_np.shape}")
    print(f"  - baseline_data_sp500.npy: 形状 {baseline_data_np.shape}")
    print(f"  - CSV文件目录: {CSV_DIR} (包含 {len(all_successful_tickers)} 个股票CSV文件)")
    
    # 验证数据质量
    print(f"\n数据质量报告 (升级版5年数据):")
    print(f"  - 时间范围: {unified_index.min().strftime('%Y-%m-%d')} 到 {unified_index.max().strftime('%Y-%m-%d')}")
    print(f"  - 交易日数: {len(unified_index)}")
    print(f"  - 特征数: {sp500_data_np.shape[2]}")
    print(f"  - 数据有效率: {np.count_nonzero(~np.isnan(sp500_data_np.sum(axis=2))) / (sp500_data_np.shape[0] * sp500_data_np.shape[1]):.2%}")
    print(f"✅ 使用升级版API：5年历史数据 (2020-2024) + 无限制调用") 
    
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