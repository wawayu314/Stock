import pandas as pd
import numpy as np

# 检查NYSE情绪数据
print("🔍 检查NYSE情绪数据...")
df = pd.read_csv('../dataset/processed_sentiment/nyse_sentiment.csv')
print(f"形状: {df.shape}")
print(f"列名前10个: {list(df.columns[:10])}")

# 获取数值列
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"数值列数量: {len(numeric_cols)}")

if len(numeric_cols) > 0:
    print("\n数值列统计信息（前5列）:")
    stats = df[numeric_cols].describe()
    print(stats.iloc[:, :5])
    
    # 检查数值范围
    print("\n数值范围检查:")
    for col in numeric_cols[:5]:
        values = df[col].dropna()
        if len(values) > 0:
            min_val = values.min()
            max_val = values.max()
            mean_val = values.mean()
            std_val = values.std()
            print(f"{col}: 范围[{min_val:.3f}, {max_val:.3f}], 均值={mean_val:.3f}, 标准差={std_val:.3f}")

print("\n✅ 检查完成") 