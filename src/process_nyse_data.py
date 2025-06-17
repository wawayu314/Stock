import numpy as np
import pickle
import os
import pandas as pd

def process_nyse_data(base_path):
    """
    加载 NYSE 数据, 对其进行清洗和归一化,
    然后保存为一套新的、干净的 .pkl, .npy, 和 .csv 文件。
    """
    print("--- 开始处理 NYSE 数据 ---")

    # 定义输入文件路径
    eod_path = os.path.join(base_path, 'eod_data.pkl')
    mask_path = os.path.join(base_path, 'mask_data.pkl')
    price_path = os.path.join(base_path, 'price_data.pkl')
    ticker_path = os.path.join(base_path, 'nyse_tickers.txt') # 读取原始 ticker 文件

    # 定义输出文件路径
    output_eod_path = os.path.join(base_path, 'eod_data_cleaned.pkl')
    output_gt_path = os.path.join(base_path, 'gt_data_cleaned.pkl')
    output_mask_path = os.path.join(base_path, 'mask_data_cleaned.pkl')
    output_price_path = os.path.join(base_path, 'price_data_cleaned.pkl')
    # 新增的输出文件
    output_nyse_npy_path = os.path.join(base_path, 'NYSE.npy')
    output_baseline_npy_path = os.path.join(base_path, 'baseline_data_nyse.npy')
    output_ticker_csv_path = os.path.join(base_path, 'nyse_ticker.csv')

    # --- 1. 加载数据 ---
    try:
        with open(eod_path, 'rb') as f:
            eod_data = pickle.load(f)
        with open(mask_path, 'rb') as f:
            mask_data = pickle.load(f)
        with open(price_path, 'rb') as f:
            price_data = pickle.load(f)
        tickers_df = pd.read_csv(ticker_path, header=None)
        all_tickers = tickers_df[0].tolist()
        print(f"成功加载原始数据。EOD 形状: {eod_data.shape}, Tickers 数量: {len(all_tickers)}")
    except Exception as e:
        print(f"错误: 加载 .pkl 或 .csv 文件失败: {e}")
        return

    # --- 2. 严格清洗数据 ---
    num_stocks = eod_data.shape[0]
    indices_to_keep = []
    
    # 设定严格的过滤标准
    MIN_PRICE = 1.0  # 任何时候价格都不能低于 $1
    MAX_PRICE = 2000.0 # 价格上限，过滤极端异常值
    MIN_VOLUME = 1000 # 最小日交易量

    print("\n开始严格清洗股票数据...")
    for i in range(num_stocks):
        stock_eod = eod_data[i, :, :]
        # O, H, L, C 价格在特征的 0, 1, 2, 4 列
        prices = stock_eod[:, [0, 1, 2, 4]]
        volumes = stock_eod[:, 3]
        
        # 检查价格和交易量是否有效
        if np.min(prices) < MIN_PRICE:
            # print(f"  剔除股票 {i}: 价格低于 ${MIN_PRICE}")
            continue
        if np.max(prices) > MAX_PRICE:
            # print(f"  剔除股票 {i}: 价格高于 ${MAX_PRICE}")
            continue
        if np.min(volumes) < MIN_VOLUME:
            # print(f"  剔除股票 {i}: 交易量低于 {MIN_VOLUME}")
            continue
        if np.any(np.isnan(stock_eod)) or np.any(np.isinf(stock_eod)):
            # print(f"  剔除股票 {i}: 包含 NaN 或 Inf")
            continue
            
        indices_to_keep.append(i)
    
    print(f"清洗完成。原始股票数: {num_stocks}, 清洗后剩余: {len(indices_to_keep)}")

    # 根据要保留的索引过滤所有数据集
    eod_data_clean = eod_data[indices_to_keep, :, :]
    mask_data_clean = mask_data[indices_to_keep, :]
    price_data_clean = price_data[indices_to_keep, :]
    tickers_clean = [all_tickers[i] for i in indices_to_keep]

    # --- 3. 数据归一化 ---
    # 模仿 "好" 数据的归一化方式：每个特征除以其各自的最大值
    print("\n开始数据归一化...")
    eod_data_normalized = np.copy(eod_data_clean)
    
    for i in range(eod_data_normalized.shape[0]): # 遍历每支股票
        for j in range(eod_data_normalized.shape[2]): # 遍历每个特征
            max_val = np.max(eod_data_normalized[i, :, j])
            if max_val > 1e-6: # 防止除以零
                eod_data_normalized[i, :, j] /= max_val
            else:
                # 如果最大值为0 (例如交易量), 则整个特征列都为0
                eod_data_normalized[i, :, j] = 0
    
    print("归一化完成。")

    # --- 4. 重新计算 GT (收益率) ---
    # 使用清洗和归一化后的收盘价计算
    print("\n重新计算收益率 (GT)...")
    price_data_normalized = eod_data_normalized[:, :, -1] # 获取归一化后的收盘价
    gt_data_recalculated = np.zeros_like(price_data_normalized)
    
    for i in range(price_data_normalized.shape[0]):
        for j in range(1, price_data_normalized.shape[1]):
            price_yesterday = price_data_normalized[i, j - 1]
            price_today = price_data_normalized[i, j]
            if price_yesterday > 1e-8: # 安全检查
                gt_data_recalculated[i, j] = (price_today - price_yesterday) / price_yesterday
            else:
                gt_data_recalculated[i, j] = 0.0

    print("收益率计算完成。")

    # --- 5. 保存新的、干净的数据 ---
    try:
        # 保存 .pkl 文件 (用于可能的调试或旧流程)
        with open(output_eod_path, 'wb') as f:
            pickle.dump(eod_data_normalized, f)
        with open(output_gt_path, 'wb') as f:
            pickle.dump(gt_data_recalculated, f)
        with open(output_mask_path, 'wb') as f:
            pickle.dump(mask_data_clean, f)
        price_data_final = eod_data_normalized[:,:,-1]
        with open(output_price_path, 'wb') as f:
            pickle.dump(price_data_final, f)
        
        # --- 6. 生成和保存额外文件 ---
        print("\n生成额外的 .npy 和 .csv 文件...")
        
        # 1. 保存 NYSE.npy
        np.save(output_nyse_npy_path, eod_data_normalized)

        # 2. 生成并保存 baseline_data_nyse.npy
        baseline_data = np.mean(eod_data_normalized, axis=0)
        np.save(output_baseline_npy_path, baseline_data)
        
        # 3. 保存 nyse_ticker.csv
        pd.DataFrame(tickers_clean).to_csv(output_ticker_csv_path, index=False, header=False)

        print(f"\n成功保存了所有文件到 {base_path}:")
        print(f"  - {os.path.basename(output_eod_path)} (pickle)")
        print(f"  - {os.path.basename(output_gt_path)} (pickle)")
        print(f"  - {os.path.basename(output_mask_path)} (pickle)")
        print(f"  - {os.path.basename(output_price_path)} (pickle)")
        print(f"  --- 新增文件 ---")
        print(f"  - {os.path.basename(output_nyse_npy_path)} (numpy array)")
        print(f"  - {os.path.basename(output_baseline_npy_path)} (numpy array)")
        print(f"  - {os.path.basename(output_ticker_csv_path)} (csv)")

    except Exception as e:
        print(f"错误: 保存清洗后的文件失败: {e}")


if __name__ == "__main__":
    process_nyse_data(base_path='../dataset/NYSE/') 