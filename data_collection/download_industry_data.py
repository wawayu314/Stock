import tushare as ts
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path

# --- 配置 ---
# 请在这里填入您的Tushare API Token
TS_TOKEN = 'ae047f06098e3b449f08cf847e3e1d17052e9da98b676489c3717e0a'
# 设置Tushare接口
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 输出文件路径
# 文件名必须是小写，以匹配train.py中的加载逻辑
OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "A_SHARE"
OUTPUT_FILE = OUTPUT_DIR / "a_share_industry_data.json"


def get_industry_classification():
    """
    获取所有A股的行业分类信息，并保存为JSON文件。
    """
    print("🚀 开始获取A股行业分类数据...")

    # 1. 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"JSON文件将保存至: {OUTPUT_FILE}")

    # 2. 直接从stock_basic接口获取行业信息
    try:
        print("⏳ 正在从'stock_basic'接口获取所有A股的行业信息...")
        # 请求包含行业信息的字段
        stock_list_df = pro.stock_basic(
            exchange='', 
            list_status='L', 
            fields='ts_code,name,industry'
        )
        
        # 移除行业信息为空的股票
        stock_list_df.dropna(subset=['industry'], inplace=True)
        stock_list_df = stock_list_df[stock_list_df['industry'] != '']
        
        print(f"✅ 成功获取 {len(stock_list_df)} 只股票的行业信息。")
    except Exception as e:
        print(f"❌ 从'stock_basic'获取股票列表和行业失败: {e}")
        return

    # 3. 将DataFrame转换为模型所需的字典格式
    # 格式: { "ts_code": {"sic_description": "industry"} }
    print("⏳ 正在将数据转换为模型所需的JSON格式...")
    all_stocks_industry = {}
    for index, row in tqdm(stock_list_df.iterrows(), total=stock_list_df.shape[0], desc="格式化数据"):
        ts_code = row['ts_code']
        industry = row['industry']
        # 创建一个包含'sic_description'键的子字典
        all_stocks_industry[ts_code] = {'sic_description': industry}
    
    print(f"✅ 数据格式转换完成。")

    # 4. 保存为JSON文件
    if all_stocks_industry:
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_stocks_industry, f, ensure_ascii=False, indent=4)
            print(f"\n🎉 行业数据成功保存到: {OUTPUT_FILE}")
            print(f"   共包含 {len(all_stocks_industry)} 条记录。")
        except Exception as e:
            print(f"❌ 保存JSON文件失败: {e}")
    else:
        print("❌ 未获取到任何行业数据，无法生成JSON文件。")


if __name__ == '__main__':
    get_industry_classification() 