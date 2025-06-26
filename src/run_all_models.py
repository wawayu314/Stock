#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主控制脚本 - 批量运行所有模型训练
功能：
1. 从config.py读取配置
2. 一次性修改所有训练脚本的数据集设置
3. 按顺序或并行运行所有模型训练
4. 管理训练流程和输出
5. 生成训练报告

使用方法:
1. 修改 config.py 中的配置
2. 运行: python src/run_all_models.py 或 cd src && python run_all_models.py
"""

import os
import re
import subprocess
import time
import threading
from datetime import datetime
import logging
import shutil
import sys

# 导入配置
try:
    from config import *
except ImportError:
    print("❌ 未找到config.py文件，使用默认配置")
    MARKET_NAME = 'NASDAQ'
    MODELS_TO_RUN = {
        'StockPredict': True, 'GCN': True, 'LSTM': True,
        'LinearModel': True, 'STHAN-SR': True
    }
    RUN_MODE = 'sequential'
    ENABLE_LOGGING = True
    LOG_DIR = '../logs'

# 模型训练脚本路径（现在都在src目录下）
TRAINING_SCRIPTS = {
    'StockPredict': 'train.py',
    'GCN': 'gcn_train.py', 
    'LSTM': 'lstm_train.py',
    'LinearModel': 'linear_train.py',
    'STHAN-SR': 'sthan_sr_train.py'
}

# 过滤掉不需要运行的模型
ACTIVE_SCRIPTS = {k: v for k, v in TRAINING_SCRIPTS.items() 
                 if MODELS_TO_RUN.get(k, False)}

# =============== 日志设置 ===============
def setup_logging():
    """设置日志系统"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f'batch_training_{MARKET_NAME}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

# =============== 文件备份和修改功能 ===============
def backup_file(file_path):
    """备份原始文件"""
    if hasattr(backup_file, '_backup_done'):
        return True
    
    try:
        backup_dir = '../backup_original_files'
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for script_path in TRAINING_SCRIPTS.values():
            if os.path.exists(script_path):
                filename = os.path.basename(script_path)
                backup_path = os.path.join(backup_dir, f"{filename}.backup_{timestamp}")
                shutil.copy2(script_path, backup_path)
                logging.info(f"📁 已备份: {script_path} -> {backup_path}")
        
        backup_file._backup_done = True
        return True
    except Exception as e:
        logging.error(f"❌ 备份失败: {str(e)}")
        return False

def modify_market_name_in_file(file_path, new_market_name):
    """修改训练脚本中的market_name设置"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换market_name设置
        pattern = r"market_name\s*=\s*['\"][^'\"]*['\"]"
        replacement = f"market_name = '{new_market_name}'"
        
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logging.info(f"✅ 已更新 {file_path} 中的market_name为: {new_market_name}")
            return True
        else:
            logging.warning(f"⚠️  在 {file_path} 中未找到market_name设置")
            return False
            
    except Exception as e:
        logging.error(f"❌ 修改 {file_path} 失败: {str(e)}")
        return False

def update_all_market_names(market_name):
    """更新所有训练脚本的market_name"""
    logging.info(f"🔄 开始更新脚本的数据集设置为: {market_name}")
    
    # 备份原始文件
    if globals().get('BACKUP_ORIGINAL_FILES', True):
        if not backup_file(None):
            logging.warning("⚠️  文件备份失败，继续执行...")
    
    success_count = 0
    for model_name, script_path in ACTIVE_SCRIPTS.items():
        if os.path.exists(script_path):
            if modify_market_name_in_file(script_path, market_name):
                success_count += 1
        else:
            logging.warning(f"⚠️  脚本文件不存在: {script_path}")
    
    logging.info(f"✅ 成功更新 {success_count}/{len(ACTIVE_SCRIPTS)} 个脚本")
    return success_count == len(ACTIVE_SCRIPTS)

# =============== 环境检查功能 ===============
def check_environment():
    """检查运行环境"""
    logging.info("🔍 检查运行环境...")
    
    issues = []
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        issues.append("Python版本过低，需要3.7+")
    
    # 检查必要的包
    required_packages = ['torch', 'numpy', 'pandas', 'openpyxl']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"缺少必要包: {package}")
    
    # 检查数据集文件（现在从src目录访问，所以是../dataset/）
    if MARKET_NAME == 'SP500':
        data_file = '../dataset/SP500/SP500.npy'
        if not os.path.exists(data_file):
            issues.append(f"SP500数据文件不存在: {data_file}")
    elif MARKET_NAME in ['NASDAQ', 'NYSE', 'A_SHARE']:
        # 适用于使用多个.pkl文件的数据集
        required_files = [
            f'../dataset/{MARKET_NAME}/eod_data.pkl',
            f'../dataset/{MARKET_NAME}/mask_data.pkl', 
            f'../dataset/{MARKET_NAME}/gt_data.pkl',
            f'../dataset/{MARKET_NAME}/price_data.pkl'
        ]
        for file_path in required_files:
            if not os.path.exists(file_path):
                issues.append(f"{MARKET_NAME}数据文件不存在: {file_path}")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logging.info(f"✅ 检测到 {gpu_count} 个GPU")
        else:
            logging.info("ℹ️  未检测到GPU，将使用CPU训练")
    except:
        pass
    
    if issues:
        logging.error("❌ 环境检查发现问题:")
        for issue in issues:
            logging.error(f"   - {issue}")
        return False
    
    logging.info("✅ 环境检查通过")
    return True

# =============== 训练执行功能 ===============
def run_single_model(model_name, script_path, log_file_path=None):
    """运行单个模型的训练脚本"""
    if not os.path.exists(script_path):
        logging.error(f"❌ 脚本文件不存在: {script_path}")
        return False, 0
    
    start_time = time.time()
    logging.info(f"🚀 开始训练 {model_name}...")
    
    try:
        # 设置GPU环境变量
        env = os.environ.copy()
        if globals().get('CUDA_VISIBLE_DEVICES'):
            env['CUDA_VISIBLE_DEVICES'] = str(globals().get('CUDA_VISIBLE_DEVICES'))
        
        # 构建命令
        cmd = [sys.executable, script_path]  # 使用当前Python解释器
        
        # 准备日志文件
        if log_file_path and globals().get('SAVE_MODEL_LOGS', True):
            model_log_file = log_file_path.replace('.log', f'_{model_name}.log')
        else:
            model_log_file = f'{LOG_DIR}/{model_name}_training.log'
        
        # 确保日志目录存在
        os.makedirs(os.path.dirname(model_log_file), exist_ok=True)
        
        # 执行训练脚本
        with open(model_log_file, 'w', encoding='utf-8') as log_f:
            process = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd(),  # 在src目录下执行
                env=env,
                timeout=globals().get('MAX_TRAINING_TIME_PER_MODEL') if globals().get('ENABLE_TIMEOUT') else None
            )
        
        elapsed_time = time.time() - start_time
        
        if process.returncode == 0:
            logging.info(f"✅ {model_name} 训练完成! 耗时: {elapsed_time:.1f}秒")
            logging.info(f"📄 详细日志: {model_log_file}")
            return True, elapsed_time
        else:
            logging.error(f"❌ {model_name} 训练失败! 退出码: {process.returncode}")
            logging.error(f"📄 错误日志: {model_log_file}")
            return False, elapsed_time
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logging.error(f"⏰ {model_name} 训练超时! (超过{globals().get('MAX_TRAINING_TIME_PER_MODEL', 3600)}秒)")
        return False, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"❌ {model_name} 运行异常: {str(e)} (耗时: {elapsed_time:.1f}秒)")
        return False, elapsed_time

def run_models_sequential(log_file_path=None):
    """顺序运行所有模型"""
    logging.info("📋 开始顺序训练所有模型...")
    
    results = {}
    times = {}
    total_start_time = time.time()
    
    for model_name, script_path in ACTIVE_SCRIPTS.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"正在训练: {model_name}")
        logging.info(f"{'='*60}")
        
        success, elapsed_time = run_single_model(model_name, script_path, log_file_path)
        results[model_name] = success
        times[model_name] = elapsed_time
        
        if not success:
            logging.warning(f"⚠️  {model_name} 训练失败，继续下一个模型...")
    
    total_time = time.time() - total_start_time
    logging.info(f"\n🏁 所有模型训练完成! 总耗时: {total_time:.1f}秒")
    
    return results, times

def run_models_parallel(log_file_path=None):
    """并行运行所有模型"""
    logging.info("🚀 开始并行训练所有模型...")
    logging.warning("⚠️  并行模式可能需要大量GPU内存!")
    
    threads = []
    results = {}
    times = {}
    total_start_time = time.time()
    
    def thread_wrapper(model_name, script_path):
        success, elapsed_time = run_single_model(model_name, script_path, log_file_path)
        results[model_name] = success
        times[model_name] = elapsed_time
    
    # 创建并启动线程
    for model_name, script_path in ACTIVE_SCRIPTS.items():
        thread = threading.Thread(
            target=thread_wrapper,
            args=(model_name, script_path),
            name=f"Thread-{model_name}"
        )
        threads.append(thread)
        thread.start()
        logging.info(f"🚀 启动 {model_name} 训练线程...")
        time.sleep(1)  # 稍微错开启动时间
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    total_time = time.time() - total_start_time
    logging.info(f"\n🏁 所有模型训练完成! 总耗时: {total_time:.1f}秒")
    
    return results, times

# =============== 结果汇总功能 ===============
def print_summary(results, times):
    """打印训练结果汇总"""
    logging.info("\n" + "="*80)
    logging.info("📊 训练结果汇总")
    logging.info("="*80)
    
    success_count = 0
    for model_name in ACTIVE_SCRIPTS.keys():
        success = results.get(model_name, False)
        elapsed_time = times.get(model_name, 0)
        status = "✅ 成功" if success else "❌ 失败"
        logging.info(f"{model_name:<15}: {status} (耗时: {elapsed_time:.1f}s)")
        if success:
            success_count += 1
    
    logging.info("-"*80)
    logging.info(f"总计: {success_count}/{len(ACTIVE_SCRIPTS)} 个模型训练成功")
    
    if success_count == len(ACTIVE_SCRIPTS):
        logging.info("🎉 所有模型训练完成！")
    else:
        logging.warning("⚠️  部分模型训练失败，请检查日志")

def check_results_file():
    """检查并显示Excel结果文件状态"""
    excel_file = '../training_results.xlsx'  # 结果文件在上级目录
    if os.path.exists(excel_file):
        file_size = os.path.getsize(excel_file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(excel_file))
        logging.info(f"📈 结果文件: {excel_file}")
        logging.info(f"   文件大小: {file_size} bytes")
        logging.info(f"   修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 尝试读取并显示简要结果
        try:
            import pandas as pd
            df = pd.read_excel(excel_file)
            logging.info(f"   记录数量: {len(df)} 条")
            if len(df) > 0:
                recent_results = df.tail(len(ACTIVE_SCRIPTS))
                logging.info("   最近结果:")
                for _, row in recent_results.iterrows():
                    logging.info(f"     {row.get('Model', 'Unknown')}: IC={row.get('Valid IC', 'N/A'):.4f}")
        except Exception as e:
            logging.info(f"   (无法读取详细内容: {e})")
    else:
        logging.warning("⚠️  未找到结果文件 ../training_results.xlsx")

# =============== 主程序 ===============
def main():
    """主程序入口"""
    # 设置日志
    if ENABLE_LOGGING:
        log_file = setup_logging()
    else:
        log_file = None
    
    logging.info("🎯 股票预测模型批量训练系统")
    logging.info("="*80)
    logging.info(f"📊 目标数据集: {MARKET_NAME}")
    logging.info(f"🔧 运行模式: {RUN_MODE}")
    logging.info(f"📝 要运行的模型: {list(ACTIVE_SCRIPTS.keys())}")
    logging.info(f"📝 总模型数量: {len(ACTIVE_SCRIPTS)}")
    logging.info(f"📁 工作目录: {os.getcwd()}")
    logging.info("="*80)
    
    # 步骤0: 环境检查
    if globals().get('CHECK_REQUIREMENTS', True):
        logging.info("\n🔍 步骤0: 环境检查")
        if not check_environment():
            logging.error("❌ 环境检查失败，终止执行")
            return
    
    # 步骤1: 更新所有脚本的数据集设置
    logging.info("\n📝 步骤1: 更新数据集设置")
    if not update_all_market_names(MARKET_NAME):
        logging.error("❌ 数据集设置更新失败，终止执行")
        return
    
    # 步骤2: 运行模型训练
    logging.info(f"\n🚀 步骤2: 开始{RUN_MODE}训练")
    time.sleep(1)  # 给用户一点时间看到信息
    
    if RUN_MODE == 'parallel':
        results, times = run_models_parallel(log_file)
    else:
        results, times = run_models_sequential(log_file)
    
    # 步骤3: 汇总结果
    logging.info("\n📊 步骤3: 汇总结果")
    print_summary(results, times)
    check_results_file()
    
    logging.info("\n🎉 批量训练任务完成!")
    if log_file:
        logging.info(f"📄 完整日志保存在: {log_file}")

if __name__ == "__main__":
    # 显示当前配置
    print("🎯 股票预测模型批量训练系统")
    print("="*60)
    print("当前配置:")
    print(f"  📊 数据集: {MARKET_NAME}")
    print(f"  🔧 运行模式: {RUN_MODE}")
    print(f"  🎯 要运行的模型: {list(ACTIVE_SCRIPTS.keys())}")
    print(f"  📝 总数量: {len(ACTIVE_SCRIPTS)}")
    print(f"  📁 工作目录: {os.getcwd()}")
    print("="*60)
    
    if len(ACTIVE_SCRIPTS) == 0:
        print("❌ 没有选择要运行的模型，请检查config.py中的MODELS_TO_RUN设置")
        sys.exit(1)
    
    # 用户确认
    try:
        user_input = input("是否继续? (y/n): ").strip().lower()
        if user_input in ['y', 'yes', '']:
            main()
        else:
            print("❌ 用户取消，退出程序")
    except KeyboardInterrupt:
        print("\n❌ 用户中断，退出程序") 