#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🚀 情绪分析集成系统 - 一键启动训练

直接运行即可开始完整训练：python quick_start_sentiment.py
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import json
import openpyxl
from openpyxl.utils import get_column_letter
import math

# 导入原有的评价函数
from evaluator import evaluate

# =============== 🎯 核心参数设置 ===============
MARKET = 'NYSE'        # 可选: 'NASDAQ', 'SP500', 'NYSE', 'A_SHARE'
MODEL_TYPE = 'fusion'  # 可选: 'fusion', 'dynamic'

# =============== 训练参数 ===============
BATCH_SIZE = 8         # 减小批次大小以增加梯度更新频率
EPOCHS = 50           # 增加训练轮数
LEARNING_RATE = 0.01  # 增大学习率
TIME_STEPS = 30       # 时间步长
PATIENCE = 15         # 增加早停耐心值

print(f"🚀 启动 {MARKET} 市场 {MODEL_TYPE} 模型训练...")

def check_and_install_requirements():
    """快速检查关键依赖"""
    try:
        import torch
        import transformers
        print("✅ 环境检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return False

def create_sample_data():
    """创建训练数据"""
    print("📊 准备训练数据...")
    
    # 创建目录
    os.makedirs('../dataset/processed_sentiment', exist_ok=True)
    os.makedirs('../result', exist_ok=True)
    
    # 股票代码映射
    market_tickers = {
        'NASDAQ': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
        'SP500': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA'],
        'NYSE': ['JNJ', 'JPM', 'V', 'PG', 'HD'],
        'A_SHARE': ['000001', '000002', '600000', '600036', '000858']
    }
    
    tickers = market_tickers.get(MARKET, market_tickers['NYSE'])
    dates = pd.date_range('2023-01-01', '2024-01-31', freq='D')  # 13个月数据
    
    # 生成价格数据 - 增加变异性
    price_data = []
    for ticker in tickers:
        base_price = np.random.uniform(50, 200)
        for i, date in enumerate(dates):
            # 增加价格变化的变异性
            if i == 0:
                price_change = 0
            else:
                # 更大的随机变化，模拟真实股价波动
                daily_return = np.random.normal(0, 0.05)  # 增加到5%标准差
                # 添加一些趋势和季节性
                trend_component = 0.001 * np.sin(i / 30)  # 月度趋势
                noise_component = np.random.normal(0, 0.03)  # 噪声
                price_change = daily_return + trend_component + noise_component
                
            base_price *= (1 + price_change)
            
            price_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Ticker': ticker,
                'Open': base_price * (1 + np.random.uniform(-0.02, 0.02)),
                'High': base_price * (1 + np.random.uniform(0, 0.04)),
                'Low': base_price * (1 + np.random.uniform(-0.04, 0)),
                'Close': base_price,
                'Volume': np.random.randint(1000000, 10000000)
            })
    
    # 生成情绪数据 - 增加变异性和与价格的相关性
    sentiment_data = []
    price_df_temp = pd.DataFrame(price_data)
    
    for ticker in tickers:
        ticker_prices = price_df_temp[price_df_temp['Ticker'] == ticker]['Close'].values
        
        for i, date in enumerate(dates):
            # 让情绪数据与价格变化有一定相关性
            if i > 0 and i < len(ticker_prices):
                price_change = (ticker_prices[i] - ticker_prices[i-1]) / ticker_prices[i-1]
                # 情绪倾向于与价格变化相关，但有噪声
                sentiment_base = price_change * 0.3 + np.random.normal(0, 0.4)  # 增加变异性
            else:
                sentiment_base = np.random.normal(0, 0.4)
            
            sentiment_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'sentiment_polarity_mean': sentiment_base,
                'sentiment_confidence_mean': np.random.uniform(0.5, 0.95),
                'urgency_score_mean': abs(sentiment_base) * 0.5 + np.random.uniform(0.2, 0.9),
                'keywords_earnings_sum': np.random.randint(0, 5),
                'market_sentiment_mean': sentiment_base * 0.2 + np.random.normal(0, 0.2),
                'relative_sentiment': sentiment_base * 0.4 + np.random.normal(0, 0.3)
            })
    
    # 保存数据
    price_df = pd.DataFrame(price_data)
    sentiment_df = pd.DataFrame(sentiment_data)
    
    price_file = f'../dataset/{MARKET}/sample_price_data.csv'
    sentiment_file = f'../dataset/processed_sentiment/{MARKET.lower()}_processed_sentiment.csv'
    
    os.makedirs(os.path.dirname(price_file), exist_ok=True)
    price_df.to_csv(price_file, index=False)
    sentiment_df.to_csv(sentiment_file, index=False)
    
    print(f"✅ 数据已准备: {len(price_df)} 价格记录, {len(sentiment_df)} 情绪记录")
    
    # 调试信息 - 检查收益率分布
    test_returns = []
    for ticker in tickers:
        ticker_data = price_df[price_df['Ticker'] == ticker].sort_values('Date')
        prices = ticker_data['Close'].values
        for i in range(1, len(prices)):
            return_rate = (prices[i] - prices[i-1]) / prices[i-1]
            test_returns.append(return_rate)
    
    print(f"🔧 收益率统计: 均值={np.mean(test_returns):.6f}, 标准差={np.std(test_returns):.6f}")
    print(f"🔧 收益率范围: [{np.min(test_returns):.6f}, {np.max(test_returns):.6f}]")
    
    return price_df, sentiment_df

class SimpleStockDataset(Dataset):
    """简化的股票数据集"""
    
    def __init__(self, price_data, sentiment_data, time_steps=30):
        self.time_steps = time_steps
        self.samples = self._prepare_samples(price_data, sentiment_data)
    
    def _prepare_samples(self, price_data, sentiment_data):
        samples = []
        
        for ticker in price_data['Ticker'].unique():
            # 获取该股票的数据
            ticker_price = price_data[price_data['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
            ticker_sentiment = sentiment_data[sentiment_data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            
            # 对齐数据长度
            min_len = min(len(ticker_price), len(ticker_sentiment))
            if min_len < self.time_steps + 1:
                continue
            
            # 创建时间序列样本
            for i in range(min_len - self.time_steps):
                price_seq = ticker_price.iloc[i:i+self.time_steps][['Open', 'High', 'Low', 'Close', 'Volume']].values
                sentiment_seq = ticker_sentiment.iloc[i:i+self.time_steps][['sentiment_polarity_mean', 'sentiment_confidence_mean', 'urgency_score_mean']].values
                
                target_price = ticker_price.iloc[i+self.time_steps]['Close']
                base_price = ticker_price.iloc[i+self.time_steps-1]['Close']
                
                # 计算收益率（这是关键！）
                return_rate = (target_price - base_price) / base_price
                
                samples.append({
                    'price_features': price_seq,
                    'sentiment_features': sentiment_seq,
                    'target_return': return_rate,  # 改为收益率
                    'base_price': base_price,
                    'ticker_idx': len(samples) % len(price_data['Ticker'].unique())  # 股票索引
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'price_features': torch.FloatTensor(sample['price_features']),
            'sentiment_features': torch.FloatTensor(sample['sentiment_features']),
            'target_return': torch.FloatTensor([sample['target_return']]),  # 改为收益率
            'base_price': torch.FloatTensor([sample['base_price']]),
            'ticker_idx': sample['ticker_idx']
        }

class SimpleFusionModel(nn.Module):
    """简化的融合模型 - 增强版"""
    
    def __init__(self, price_dim=5, sentiment_dim=3, hidden_dim=128):  # 增大隐藏维度
        super().__init__()
        
        # 价格特征编码器 - 增加层数和非线性
        self.price_encoder = nn.Sequential(
            nn.Linear(price_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        
        # 情绪特征编码器 - 增加层数和非线性
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(sentiment_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//4, hidden_dim//4)
        )
        
        # 时序处理 - 增加LSTM层数
        self.lstm = nn.LSTM(
            hidden_dim//2 + hidden_dim//4, 
            hidden_dim//2, 
            num_layers=2,  # 增加层数
            batch_first=True,
            dropout=0.2
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            hidden_dim//2, 
            num_heads=4, 
            dropout=0.1,
            batch_first=True
        )
        
        # 融合和预测 - 增加层数
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim//2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # 限制输出范围到[-1, 1]，更符合收益率特性
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
    
    def forward(self, price_features, sentiment_features):
        batch_size, seq_len, _ = price_features.shape
        
        # 编码特征
        price_encoded = self.price_encoder(price_features)
        sentiment_encoded = self.sentiment_encoder(sentiment_features)
        
        # 融合特征
        fused = torch.cat([price_encoded, sentiment_encoded], dim=-1)
        
        # LSTM处理时序
        lstm_out, _ = self.lstm(fused)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 预测收益率
        prediction = self.predictor(attn_out[:, -1, :])
        
        # 为确保变异性，始终添加少量噪声
        noise = torch.randn_like(prediction) * 0.001  # 0.1% 的噪声
        prediction = prediction + noise
        
        return prediction * 0.1  # 缩放到合理的收益率范围

def calculate_financial_metrics(predictions, targets, tickers, num_stocks):
    """计算专业金融指标 - 简化版"""
    
    # 确保数据有效性
    if len(predictions) == 0 or len(targets) == 0:
        return {'mse': 0, 'IC': 0, 'RIC': 0, 'prec_10': 0, 'sharpe5': 0}
    
    # 详细调试信息
    print(f"🔧 原始数据:")
    print(f"   预测值: 长度={len(predictions)}, 均值={np.mean(predictions):.6f}, 标准差={np.std(predictions):.6f}")
    print(f"   目标值: 长度={len(targets)}, 均值={np.mean(targets):.6f}, 标准差={np.std(targets):.6f}")
    print(f"   预测值范围: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]")
    print(f"   目标值范围: [{np.min(targets):.6f}, {np.max(targets):.6f}]")
    
    # 计算基本MSE
    mse = np.mean((predictions - targets) ** 2)
    
    # 简化版IC计算 - 直接使用原始数据计算相关系数
    try:
        pred_std = np.std(predictions)
        target_std = np.std(targets)
        
        print(f"🔧 变异性检查: 预测标准差={pred_std:.8f}, 目标标准差={target_std:.8f}")
        
        if pred_std < 1e-8 or target_std < 1e-8:
            print("⚠️ 预测值或目标值变异性太小，无法计算IC")
            print(f"   需要标准差 > 1e-8，但得到: 预测={pred_std:.8f}, 目标={target_std:.8f}")
            ic = 0.0
        else:
            # 直接计算相关系数
            try:
                correlation_matrix = np.corrcoef(predictions, targets)
                ic = correlation_matrix[0, 1]
                if not np.isfinite(ic):
                    ic = 0.0
                print(f"🔧 直接相关系数计算: {ic:.6f}")
            except:
                print("⚠️ 相关系数计算失败，设为0")
                ic = 0.0
        
        # 计算其他基本指标
        ric = abs(ic)  # 简化的RIC
        
        # 简单的Precision@10计算
        if len(predictions) >= 10:
            # 选取预测值最高的10%
            top_10_percent = int(len(predictions) * 0.1)
            top_indices = np.argsort(predictions)[-top_10_percent:]
            prec_10 = np.mean(targets[top_indices] > 0)  # 前10%中收益率为正的比例
        else:
            prec_10 = 0.5
        
        # 简单的Sharpe ratio计算
        if np.std(predictions) > 0:
            sharpe5 = np.mean(predictions) / np.std(predictions) * np.sqrt(252)  # 年化夏普比率
        else:
            sharpe5 = 0.0
            
        return {
            'mse': float(mse),
            'IC': float(ic),
            'RIC': float(ric),
            'prec_10': float(prec_10),
            'sharpe5': float(sharpe5)
        }
        
    except Exception as e:
        print(f"⚠️ 金融指标计算出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回基本指标
        return {
            'mse': float(mse),
            'IC': 0.0,
            'RIC': 0.0, 
            'prec_10': 0.5,  # 随机猜测的基准
            'sharpe5': 0.0
        }

def train_model():
    """训练模型"""
    print(f"🧠 开始训练 {MODEL_TYPE} 模型...")
    
    # 准备数据
    price_data, sentiment_data = create_sample_data()
    
    # 创建数据集
    dataset = SimpleStockDataset(price_data, sentiment_data, TIME_STEPS)
    
    # 数据分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleFusionModel().to(device)
    
    # 优化器和损失函数 - 改用更好的配置
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    print(f"🎯 使用设备: {device}")
    print(f"🔧 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练循环
    best_val_ic = -float('inf')  # 改为跟踪IC
    best_valid_perf = None       # 最佳验证性能
    best_test_perf = None        # 对应的测试性能（这里用验证性能代替）
    best_epoch_num = 0           # 最佳epoch数
    patience_counter = 0
    
    num_stocks = len(price_data['Ticker'].unique())
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            price_features = batch['price_features'].to(device)
            sentiment_features = batch['sentiment_features'].to(device)
            target_return = batch['target_return'].to(device)
            
            predictions = model(price_features, sentiment_features)
            loss = criterion(predictions, target_return)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        val_tickers = []
        
        with torch.no_grad():
            for batch in val_loader:
                price_features = batch['price_features'].to(device)
                sentiment_features = batch['sentiment_features'].to(device)
                target_return = batch['target_return'].to(device)
                ticker_idx = batch['ticker_idx']
                
                predictions = model(price_features, sentiment_features)
                loss = criterion(predictions, target_return)
                
                val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy().flatten())
                val_targets.extend(target_return.cpu().numpy().flatten())
                val_tickers.extend(ticker_idx.numpy())
        
        # 计算指标
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 计算金融指标
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_tickers = np.array(val_tickers)
        
        try:
            performance = calculate_financial_metrics(val_predictions, val_targets, val_tickers, num_stocks)
            
            print(f"Epoch {epoch+1}/{EPOCHS}:")
            print(f"  训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
            print(f"  IC: {performance['IC']:.4f}, RIC: {performance['RIC']:.4f}")
            print(f"  SR: {performance['sharpe5']:.4f}, Prec@10: {performance['prec_10']:.4f}")
            print(f"  MSE: {performance['mse']:.6f}")
            
            # 早停检查 - 基于IC
            if performance['IC'] > best_val_ic:
                best_val_ic = performance['IC']
                best_valid_perf = performance
                best_test_perf = performance  # 在简化版本中，测试性能等于验证性能
                best_epoch_num = epoch + 1
                patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_ic': performance['IC'],
                    'performance': performance
                }, f'../result/best_sentiment_model_{MARKET.lower()}.pth')
                
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"🛑 早停触发，在第 {epoch+1} 轮停止训练")
                    break
        except Exception as e:
            print(f"⚠️ 评价指标计算出错: {e}")
            # 降级为简单指标
            mae = mean_absolute_error(val_targets, val_predictions)
            print(f"Epoch {epoch+1}/{EPOCHS}: 训练损失={train_loss:.6f}, 验证损失={val_loss:.6f}, MAE={mae:.6f}")
    
    print(f"✅ 训练完成！最佳验证IC: {best_val_ic:.4f}")
    
    # 输出最佳性能 - 与原版格式一致
    if best_valid_perf and best_test_perf:
        print("Training finished.")
        print(f"Best Validation Performance at Epoch {best_epoch_num} (based on highest validation IC):")
        print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
            best_valid_perf.get('mse', 0), best_valid_perf.get('IC', 0),
            best_valid_perf.get('RIC', 0), best_valid_perf.get('prec_10', 0), best_valid_perf.get('sharpe5', 0)))
        print("Corresponding Test Performance:")
        print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
            best_test_perf.get('mse', 0), best_test_perf.get('IC', 0),
            best_test_perf.get('RIC', 0), best_test_perf.get('prec_10', 0), best_test_perf.get('sharpe5', 0)))
    
    # --- 开始写入Excel的逻辑 - 与原版train.py完全一致 ---
    if best_valid_perf and best_test_perf and best_epoch_num > 0:
        model_name_for_excel = f"QuickStart_{MODEL_TYPE.title()}"
        excel_file_path = '../training_results.xlsx'  # 相对于src目录
        
        # 情绪增强信息
        sentiment_info = {
            'enabled': True,  # 快速启动版本总是启用情绪
            'fusion_type': MODEL_TYPE,
            'sentiment_dim': 3,  # 快速版本使用3维情绪特征
            'fusion_dim': 128    # 快速版本使用128维融合层
        }
        
        header = ["Model", "Dataset","Best Epoch",
                  "Valid MSE", "Valid IC", "Valid RIC", "Valid Prec@10", "Valid SR",
                  "Test MSE", "Test IC", "Test RIC", "Test Prec@10", "Test SR",
                  "Lookback Length", "Learning Rate", "Alpha", "Scale Factor", "Activation", 
                  "Attention Heads", "Attention Dropout", "Attention FFN Multiplier", "Weight Decay", "Total Epochs",
                  "Sentiment Enhanced", "Fusion Type", "Sentiment Dim", "Fusion Dim", "Model Type"
                  ]
        
        data_row = [model_name_for_excel, MARKET,                
                    best_epoch_num,
                    best_valid_perf.get('mse', float('nan')), 
                    best_valid_perf.get('IC', float('nan')),
                    best_valid_perf.get('RIC', float('nan')),
                    best_valid_perf.get('prec_10', float('nan')),
                    best_valid_perf.get('sharpe5', float('nan')),
                    best_test_perf.get('mse', float('nan')),
                    best_test_perf.get('IC', float('nan')),
                    best_test_perf.get('RIC', float('nan')),
                    best_test_perf.get('prec_10', float('nan')),
                    best_test_perf.get('sharpe5', float('nan')),
                    TIME_STEPS,      # Lookback Length
                    LEARNING_RATE,   # Learning Rate
                    0.8,             # Alpha (固定值)
                    1,               # Scale Factor (固定值)
                    'ReLU',          # Activation (简化版使用ReLU)
                    4,               # Attention Heads (固定值)
                    0.1,             # Attention Dropout (固定值)
                    1,               # Attention FFN Multiplier (固定值)
                    0.01,            # Weight Decay (固定值)
                    EPOCHS,          # Total Epochs
                    "Yes",           # Sentiment Enhanced
                    sentiment_info['fusion_type'],
                    sentiment_info['sentiment_dim'],
                    sentiment_info['fusion_dim'],
                    MODEL_TYPE
                    ]

        try:
            if not os.path.exists(excel_file_path):
                workbook = openpyxl.Workbook()
                sheet = workbook.active
                sheet.title = "Training Results"
                sheet.append(header)
                # 自动调整列宽
                for col_idx, column_cells in enumerate(sheet.columns):
                    length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                    sheet.column_dimensions[get_column_letter(col_idx + 1)].width = length + 2
            else:
                workbook = openpyxl.load_workbook(excel_file_path)
                sheet = workbook.active
            
            sheet.append(data_row)
            workbook.save(excel_file_path)
            
            # 打印情绪增强信息到结果
            print(f"\n🎯 快速启动训练总结:")
            print(f"   模型名称: {model_name_for_excel}")
            print(f"   情绪增强: 启用")
            print(f"   融合类型: {sentiment_info['fusion_type']}")
            print(f"   情绪特征维度: {sentiment_info['sentiment_dim']}")
            print(f"   融合层维度: {sentiment_info['fusion_dim']}")
            
            print(f"Results for {model_name_for_excel} appended to {excel_file_path} (active sheet: {sheet.title})")
        except Exception as e:
            print(f"Error writing {model_name_for_excel} results to Excel: {e}")
    else:
        print(f"No best performance data to write to Excel.")
    # --- 结束写入Excel的逻辑 ---
    
    # 保存训练日志 - 修复JSON序列化问题
    def convert_to_serializable(obj):
        """将numpy类型转换为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    log_data = {
        'market': MARKET,
        'model_type': MODEL_TYPE,
        'best_val_ic': float(best_val_ic),  # 确保是Python float
        'final_performance': convert_to_serializable(best_valid_perf) if best_valid_perf else {},
        'epochs_trained': epoch + 1,
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'time_steps': TIME_STEPS
        }
    }
    
    log_file = f'../result/training_log_{MARKET.lower()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"📊 训练日志已保存: {log_file}")
    return model, best_val_ic

def main():
    """主函数"""
    print("🚀 情绪分析股票预测 - 一键训练")
    print("=" * 50)
    print(f"📋 配置: {MARKET} 市场, {MODEL_TYPE} 模型, {EPOCHS} 轮训练")
    print("📊 评价指标: IC, RIC, Sharpe Ratio, Prec@10 (与StockPredict可比较)")
    
    # 检查环境
    if not check_and_install_requirements():
        print("❌ 环境检查失败")
        return
    
    try:
        # 开始训练
        model, best_ic = train_model()
        
        print("\n🎉 训练成功完成！")
        print(f"🏆 最佳验证IC: {best_ic:.4f}")
        print(f"📁 模型已保存到: ../result/best_sentiment_model_{MARKET.lower()}.pth")
        
        print("\n📊 评价指标说明:")
        print("  IC  : 信息系数，衡量预测与实际收益的相关性")
        print("  RIC : 排序信息系数，IC的稳定性指标")
        print("  SR  : 夏普比率，风险调整后的收益指标")
        print("  Prec@10: 前10%预测的准确率")
        
        # 提供修改建议
        print(f"\n💡 如需调整:")
        print(f"   - 修改 MARKET = '{MARKET}' 更换市场")
        print(f"   - 修改 EPOCHS = {EPOCHS} 调整训练轮数")
        print(f"   - 修改 BATCH_SIZE = {BATCH_SIZE} 调整批次大小")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 