import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
import json
import sys

# 将项目根目录添加到 sys.path
# 这确保了无论从哪里运行脚本，`data_collection` 模块都可以被找到。
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data_collection.process_downloaded_data import process_market_data

# --- CONFIGURATION ---
API_KEY = "NoJWwFDrkW3bh0oyp1Wc8ZhYjLu_0zZX"
START_DATE = "2000-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')
BASE_DIR = Path("dataset_raw") # 将数据保存到项目根目录下的 dataset_raw 文件夹
# Pro plan allows high concurrency. Adjust if needed.
# 设置为200个并发请求以实现高速下载。
CONCURRENCY_LIMIT = 200
# 主要美国交易所: NYSE, NASDAQ, BATS, ARCA
TARGET_EXCHANGES = [
    # NYSE Group
    "XNYS", "AMEX", "ARCA", "XCHI", "XCIS",
    # Nasdaq Group
    "XNAS", "XBOS", "XPSX", "XPHL",
    # Cboe Group
    "BATS", "BYX", "EDGX", "EDGA",
    # MIAX Group
    "MPRL", "MRLD", "MXAM",
    # Others
    "MEMX", "IEXG", "LTSE"
]

async def get_all_tickers(session: aiohttp.ClientSession, exchanges: list) -> list:
    """从指定的交易所获取所有活跃的股票代码。"""
    tickers = []
    exchange_set = set(exchanges)
    
    url = f"https://api.polygon.io/v3/reference/tickers?active=true&market=stocks&limit=1000&apiKey={API_KEY}"

    print("🚀 开始获取所有股票代码...")
    
    pbar = tqdm_asyncio(desc="获取股票代码...")
    while url:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                
                for ticker_info in data.get("results", []):
                    if ticker_info.get("primary_exchange") in exchange_set:
                        tickers.append((ticker_info["ticker"], ticker_info["primary_exchange"]))

                pbar.update(len(data.get("results", [])))
                url = data.get("next_url")
                if url:
                    url += f"&apiKey={API_KEY}" # Polygon的next_url不包含API密钥，需要手动添加

        except aiohttp.ClientError as e:
            print(f"❌ 获取股票代码列表时发生网络错误: {e}. 5秒后重试...")
            await asyncio.sleep(5)
    
    pbar.close()
    # 去重
    unique_tickers = list(set(tickers))
    print(f"✅ 股票代码获取完成！共发现 {len(unique_tickers)} 个符合条件的股票。")
    return unique_tickers

async def fetch_daily_data(session: aiohttp.ClientSession, ticker: str, exchange: str, semaphore: asyncio.Semaphore) -> None:
    """为单个股票代码获取并保存日线EOD数据。"""
    async with semaphore:
        save_dir = BASE_DIR / exchange
        save_path = save_dir / f"{ticker}.csv"
        
        # 确保在保存文件前，目录已存在
        save_dir.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            return

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{START_DATE}/{END_DATE}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
        
        try:
            async with session.get(url) as response:
                if response.status == 404: # Not found, skip
                    return
                response.raise_for_status()
                data = await response.json()

                if data.get("resultsCount", 0) > 0 and "results" in data:
                    df = pd.DataFrame(data["results"])
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                    df = df[['date', 'o', 'h', 'l', 'c', 'v']]
                    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                    
                    df.to_csv(save_path, index=False)
        except aiohttp.ClientError:
            pass
        except Exception:
            pass

async def fetch_news_data(session: aiohttp.ClientSession, ticker: str, exchange: str, semaphore: asyncio.Semaphore) -> None:
    """获取并保存单个股票的所有相关新闻。"""
    async with semaphore:
        save_dir = BASE_DIR / exchange / "news"
        save_path = save_dir / f"{ticker}_news.json"

        # 确保在保存文件前，目录已存在
        save_dir.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            return

        all_news = []
        news_url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=1000&apiKey={API_KEY}"
        
        try:
            while news_url:
                async with session.get(news_url) as response:
                    if response.status == 404:
                        break
                    response.raise_for_status()
                    data = await response.json()
                    
                    all_news.extend(data.get("results", []))
                    
                    news_url = data.get("next_url")
                    if news_url:
                        news_url += f"&apiKey={API_KEY}"

            if all_news:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(all_news, f, indent=4)
        except (aiohttp.ClientError, Exception):
            pass

async def fetch_ticker_details(session: aiohttp.ClientSession, ticker: str, exchange: str, semaphore: asyncio.Semaphore) -> None:
    """获取并保存单个股票的详细信息（用于后续聚合）。"""
    async with semaphore:
        save_dir = BASE_DIR / exchange / "details"
        save_path = save_dir / f"{ticker}_details.json"

        # 确保在保存文件前，目录已存在
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if save_path.exists():
            return

        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={API_KEY}"
        try:
            async with session.get(url) as response:
                if response.status == 404:
                    return
                response.raise_for_status()
                data = await response.json()

                if "results" in data:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(data["results"], f, indent=4)
        except (aiohttp.ClientError, Exception):
            pass

def aggregate_details_data_by_market():
    """按交易所分别聚合所有下载的股票详细信息。"""
    print("\n" + "="*60)
    print("⚙️  开始按交易所聚合行业与公司详细数据...")

    for exchange in TARGET_EXCHANGES:
        print(f"\n--- 正在处理交易所: {exchange} ---")
        exchange_dir = BASE_DIR / exchange
        details_paths = list(exchange_dir.glob("details/*.json"))

        if not details_paths:
            print(f"⚠️ 在 {exchange} 交易所未找到任何详细数据文件，跳过。")
            continue

        all_details = {}
        for path in tqdm(details_paths, desc=f"聚合 {exchange} 数据"):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    details = json.load(f)
                    ticker = details.get("ticker")
                    if ticker:
                        all_details[ticker] = details
                except json.JSONDecodeError:
                    print(f"⚠️ 文件 {path} JSON 解析失败，已跳过。")
                    continue
        
        if not all_details:
            print(f"⚠️ 在 {exchange} 没有可聚合的数据。")
            continue

        # 保存JSON
        json_path = BASE_DIR / f"{exchange}_industry_data.json"
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(all_details, f, indent=4)
        print(f"✅ {exchange} 交易所的公司详细数据已聚合到: {json_path}")

        # 保存CSV
        try:
            df = pd.DataFrame.from_dict(all_details, orient='index')
            # 选择一些关键列进行展平
            if 'name' in df.columns:
                df['name'] = df['name'].str.title()
            
            df['sic_code'] = df.get('sic_code')
            df['sic_description'] = df.get('sic_description')
            df['market_cap'] = df.get('market_cap')
            df['total_employees'] = df.get('total_employees')
            
            # 提取地址信息
            if 'address' in df.columns:
                address_df = pd.json_normalize(df['address'].fillna({}))
                address_df.columns = [f"address_{col}" for col in address_df.columns]
                # 合并
                df = pd.concat([df.drop('address', axis=1), address_df], axis=1)

            csv_columns = ['ticker', 'name', 'primary_exchange', 'market', 'locale', 'type', 'currency_name', 'market_cap', 'sic_code', 'sic_description', 'total_employees', 'homepage_url', 'address_address1', 'address_city', 'address_state', 'address_postal_code']
            # 确保所有列都存在
            for col in csv_columns:
                if col not in df.columns:
                    df[col] = None

            csv_path = BASE_DIR / f"{exchange}_industry_data.csv"
            df[csv_columns].to_csv(csv_path)
            print(f"✅ {exchange} 交易所的公司详细数据已保存为CSV: {csv_path}")
        except Exception as e:
            print(f"❌ 为 {exchange} 创建CSV文件失败: {e}")

async def main():
    """主函数，用于协调数据下载过程。"""
    print("="*60)
    print("        Polygon.io 全量数据下载器 (Pro 版本)")
    print("="*60)
    print(f"API 密钥: ...{API_KEY[-4:]}")
    print(f"目标交易所: {', '.join(TARGET_EXCHANGES)}")
    print(f"数据时间范围: {START_DATE} to {END_DATE}")
    print(f"数据保存至: {BASE_DIR.resolve()}")
    print(f"最大并发请求数: {CONCURRENCY_LIMIT}")
    print("="*60)
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        all_tickers = await get_all_tickers(session, TARGET_EXCHANGES)
        
        if not all_tickers:
            print("❌ 未能获取任何股票代码，程序即将退出。")
            return

        print(f"\n🚀 开始为 {len(all_tickers)} 只股票下载 日线、新闻和详细数据...")
        
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        
        eod_tasks = [fetch_daily_data(session, ticker, ex, semaphore) for ticker, ex in all_tickers]
        news_tasks = [fetch_news_data(session, ticker, ex, semaphore) for ticker, ex in all_tickers]
        details_tasks = [fetch_ticker_details(session, ticker, ex, semaphore) for ticker, ex in all_tickers]
        
        all_tasks = eod_tasks + news_tasks + details_tasks
        
        await tqdm_asyncio.gather(*all_tasks, desc="下载全部数据")

    # --- 数据后处理 ---
    aggregate_details_data_by_market()

    # --- 运行新的数据处理流程 ---
    print("\n" + "="*60)
    print("🚀 开始运行数据预处理流程...")
    print("="*60)
    for exchange in TARGET_EXCHANGES:
        process_market_data(market_name=exchange, base_dir=BASE_DIR)

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print("🎉🎉🎉 全部数据处理完成! 🎉🎉🎉")
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"数据已保存在 '{BASE_DIR.resolve()}' 目录中，并按交易所分类。")
    print(f"聚合后的行业数据文件以及处理后的模型数据都已生成。")
    print("="*60)

if __name__ == "__main__":
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ 操作被用户中断。") 