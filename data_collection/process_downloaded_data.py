import numpy as np
import pickle
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def process_market_data(market_name: str, base_dir: Path, trading_days: int = 252, num_features: int = 5):
    """
    处理单个市场的原始CSV数据，生成用于模型训练的 .pkl 和 .npy 文件。
    
    Args:
        market_name (str): 交易所名称 (例如 "XNYS", "XNAS").
        base_dir (Path): 数据集的根目录 (例如 "dataset_raw").
        trading_days (int): 我们期望的交易日数。
        num_features (int): 特征数量 (open, high, low, volume, close).
    """
    print("\n" + "="*60)
    print(f"⚙️  开始处理市场: {market_name}")
    print("="*60)
    
    market_path = base_dir / market_name
    csv_files = list(market_path.glob("*.csv"))
    
    if not csv_files:
        print(f"⚠️ 在 {market_path} 中没有找到任何CSV文件，跳过。")
        return

    all_tickers = [f.stem for f in csv_files]
    num_stocks = len(all_tickers)
    
    # 初始化空的 numpy 数组
    eod_data = np.zeros((num_stocks, trading_days, num_features))
    mask_data = np.ones((num_stocks, trading_days)) # 初始假设所有数据都存在
    price_data = np.zeros((num_stocks, trading_days))

    # --- 1. 从CSV加载数据到Numpy数组 ---
    print(f"⏳ 正在从 {len(csv_files)} 个CSV文件加载数据...")
    valid_tickers = []
    valid_stock_indices = []

    for i, ticker in enumerate(tqdm(all_tickers, desc=f"加载 {market_name} 数据")):
        file_path = market_path / f"{ticker}.csv"
        df = pd.read_csv(file_path)
        
        # 数据对齐和填充
        df['date'] = pd.to_datetime(df['date'])
        # 截取最近的 trading_days 天数据
        df = df.tail(trading_days)
        
        if len(df) < trading_days:
            # 如果数据不足，我们将其标记为无效并跳过
            mask_data[i, :] = 0 # 标记所有天数为缺失
            continue

        # 提取特征
        # 顺序: open, high, low, volume, close
        stock_features = df[['open', 'high', 'low', 'volume', 'close']].values
        
        eod_data[i, :, :] = stock_features
        price_data[i, :] = df['close'].values
        valid_tickers.append(ticker)
        valid_stock_indices.append(i)

    # 只保留那些有完整数据的股票
    eod_data = eod_data[valid_stock_indices]
    mask_data = mask_data[valid_stock_indices]
    price_data = price_data[valid_stock_indices]
    
    if len(valid_tickers) == 0:
        print(f"❌ 在 {market_name} 市场没有找到任何拥有足够数据的股票。")
        return
        
    print(f"✅ 数据加载完成。发现 {len(valid_tickers)} 只拥有足够历史数据的股票。")

    # --- 2. 严格清洗数据 ---
    indices_to_keep = []
    num_stocks = eod_data.shape[0]
    
    MIN_PRICE = 1.0
    MAX_PRICE = 2000.0
    MIN_VOLUME = 1000

    print("\n⏳ 开始严格清洗股票数据...")
    for i in range(num_stocks):
        stock_eod = eod_data[i, :, :]
        # O, H, L, C 价格在特征的 0, 1, 2, 4 列
        prices = stock_eod[:, [0, 1, 2, 4]]
        volumes = stock_eod[:, 3]
        
        if np.min(prices) < MIN_PRICE or np.max(prices) > MAX_PRICE or np.min(volumes) < MIN_VOLUME or np.any(np.isnan(stock_eod)) or np.any(np.isinf(stock_eod)):
            continue
        indices_to_keep.append(i)
    
    print(f"✅ 清洗完成。原始股票数: {num_stocks}, 清洗后剩余: {len(indices_to_keep)}")

    eod_data_clean = eod_data[indices_to_keep]
    mask_data_clean = mask_data[indices_to_keep]
    price_data_clean = price_data[indices_to_keep]
    tickers_clean = [valid_tickers[i] for i in indices_to_keep]

    # --- 3. 数据归一化 ---
    print("\n⏳ 开始数据归一化...")
    eod_data_normalized = np.copy(eod_data_clean)
    
    for i in range(eod_data_normalized.shape[0]):
        for j in range(eod_data_normalized.shape[2]):
            max_val = np.max(eod_data_normalized[i, :, j])
            if max_val > 1e-6:
                eod_data_normalized[i, :, j] /= max_val
    
    print("✅ 归一化完成。")

    # --- 4. 重新计算 GT (收益率) ---
    print("\n⏳ 重新计算收益率 (GT)...")
    price_data_normalized = eod_data_normalized[:, :, -1]
    gt_data_recalculated = np.zeros_like(price_data_normalized)
    
    for i in range(price_data_normalized.shape[0]):
        for j in range(1, price_data_normalized.shape[1]):
            price_yesterday = price_data_normalized[i, j - 1]
            price_today = price_data_normalized[i, j]
            if price_yesterday > 1e-8:
                gt_data_recalculated[i, j] = (price_today - price_yesterday) / price_yesterday

    print("✅ 收益率计算完成。")

    # --- 5. 保存新的、干净的数据 ---
    output_dir = base_dir / market_name / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("\n💾 正在保存处理后的文件...")
        # 定义输出路径
        eod_path = output_dir / 'eod_data.pkl'
        gt_path = output_dir / 'gt_data.pkl'
        mask_path = output_dir / 'mask_data.pkl'
        price_path = output_dir / 'price_data.pkl'
        market_npy_path = output_dir / f'{market_name}.npy'
        ticker_csv_path = output_dir / f'{market_name}_tickers.csv'

        # 保存 Pickle 文件
        with open(eod_path, 'wb') as f: pickle.dump(eod_data_normalized, f)
        with open(gt_path, 'wb') as f: pickle.dump(gt_data_recalculated, f)
        with open(mask_path, 'wb') as f: pickle.dump(mask_data_clean, f)
        with open(price_path, 'wb') as f: pickle.dump(eod_data_normalized[:,:,-1], f)
        
        # 保存 Numpy 文件
        np.save(market_npy_path, eod_data_normalized)
        
        # 保存 Ticker 列表
        pd.DataFrame(tickers_clean).to_csv(ticker_csv_path, index=False, header=False)

        print(f"\n🎉 成功为 {market_name} 保存了所有文件到 {output_dir}")

    except Exception as e:
        print(f"❌ 错误: 保存清洗后的文件失败: {e}")

if __name__ == '__main__':
    # 这是一个示例，展示如何独立运行此脚本
    # 假设数据已经通过 download_all_data.py 下载
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "dataset_raw"
    
    # 要处理的市场列表
    exchanges_to_process = ["A_SHARE"]
    
    for exchange in exchanges_to_process:
        if (raw_data_dir / exchange).exists():
            process_market_data(market_name=exchange, base_dir=raw_data_dir)
        else:
            print(f"目录 {raw_data_dir / exchange} 不存在，跳过。") 