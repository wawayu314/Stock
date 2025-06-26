import tushare as ts
import pandas as pd
import os
import time
from tqdm import tqdm
from pathlib import Path

# --- 配置 ---
# 请在这里填入您的Tushare API Token
TS_TOKEN = 'ae047f06098e3b449f08cf847e3e1d17052e9da98b676489c3717e0a'
# 设置Tushare接口
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 数据获取时间范围
START_DATE = '20190101'
END_DATE = '20240520' # 您可以根据需要修改

# 数据保存路径
OUTPUT_DIR = Path(__file__).parent.parent / "dataset_raw" / "A_SHARE"

# API调用间隔，防止频率超限 (单位: 秒)
API_DELAY = 0.15

def download_all_ashare_data():
    """
    下载所有A股在指定时间范围内的日线数据。
    """
    print("🚀 开始下载A股数据...")
    
    # 1. 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"数据将保存至: {OUTPUT_DIR}")

    # 2. 获取A股所有上市公司列表
    try:
        print("获取所有A股上市公司列表中...")
        stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
        print(f"✅ 成功获取 {len(stock_list)} 只A股股票。")
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")
        return

    # 3. 遍历每只股票，下载数据并保存
    print(f"⏳ 开始下载从 {START_DATE} 到 {END_DATE} 的日线数据...")
    
    for index, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="下载进度"):
        ts_code = row['ts_code']
        output_path = OUTPUT_DIR / f"{ts_code}.csv"

        # 如果文件已存在，则跳过
        if output_path.exists():
            continue

        try:
            # 使用 pro_bar 获取后复权日线数据
            # 字段: trade_date, open, high, low, close, vol
            df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=START_DATE, end_date=END_DATE,
                            fields='trade_date,open,high,low,close,vol')
            
            if df is None or df.empty:
                # print(f"⚠️ 未获取到 {ts_code} 的数据，跳过。")
                continue

            # 调整列名以匹配处理脚本的格式
            # trade_date -> date, vol -> volume
            df.rename(columns={'trade_date': 'date', 'vol': 'volume'}, inplace=True)
            
            # 反转数据，使其按时间升序排列
            df = df.iloc[::-1].reset_index(drop=True)

            # 保存为CSV
            df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"❌ 下载 {ts_code} 数据时出错: {e}")
        
        # 暂停一下，避免API限流
        time.sleep(API_DELAY)
        
    print("\n🎉 全部数据下载完成！")

if __name__ == '__main__':
    download_all_ashare_data() 