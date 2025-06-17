#!/usr/bin/env python3
"""
统一的新闻数据收集管理脚本
用于管理NASDAQ、SP500、NYSE三个市场的新闻数据收集
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, market_name):
    """运行单个数据收集脚本"""
    print(f"\n{'='*80}")
    print(f"🚀 开始收集 {market_name} 新闻数据")
    print(f"脚本: {script_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    try:
        # 运行脚本
        result = subprocess.run([sys.executable, script_name], 
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=False,
                              text=True)
        
        if result.returncode == 0:
            print(f"\n✅ {market_name} 新闻数据收集完成！")
        else:
            print(f"\n❌ {market_name} 新闻数据收集失败，退出代码: {result.returncode}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"\n❌ 运行 {script_name} 时出错: {e}")
        return False

def show_menu():
    """显示菜单选项"""
    print(f"\n📰 新闻数据收集管理系统 (升级版API - 高速模式)")
    print(f"{'='*60}")
    print("1. 收集 NASDAQ 新闻数据 (高速模式)")
    print("2. 收集 S&P 500 新闻数据 (高速模式)") 
    print("3. 收集 NYSE 新闻数据 (高速模式)")
    print("4. 收集所有市场新闻数据 (NASDAQ + SP500 + NYSE - 高速模式)")
    print("5. 查看数据收集状态")
    print("6. 安装依赖包")
    print("0. 退出")
    print(f"{'='*60}")
    print("⚡ 升级版API特性：大批次 + 短间隔 + 无限制调用")

def check_dependencies():
    """检查依赖包是否已安装"""
    required_packages = ['textblob', 'pandas', 'numpy', 'requests', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """安装依赖包"""
    print("🔧 正在安装必要的依赖包...")
    
    try:
        # 安装requirements_sentiment.txt中的包
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_sentiment.txt'], 
                      check=True)
        
        # 下载nltk数据
        print("📥 下载 NLTK 数据...")
        subprocess.run([sys.executable, '-c', 
                       'import nltk; nltk.download("punkt"); nltk.download("brown")'], 
                      check=True)
        
        print("✅ 依赖包安装完成！")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装依赖包失败: {e}")
        return False

def check_data_status():
    """检查各个市场的数据收集状态"""
    print(f"\n📊 数据收集状态检查")
    print(f"{'='*60}")
    
    markets = {
        'NASDAQ': ('../dataset/NASDAQ/news_sentiment/', 952),
        'SP500': ('../dataset/SP500/news_sentiment/', 502),
        'NYSE': ('../dataset/NYSE/news_sentiment/', 1619)
    }
    
    total_expected = sum(count for _, count in markets.values())
    total_collected = 0
    
    for market, (path, expected_count) in markets.items():
        print(f"\n🏢 {market} 市场:")
        
        abs_path = os.path.abspath(path)
        
        if os.path.exists(abs_path):
            # 检查文件
            files = os.listdir(abs_path)
            
            # 统计不同类型的文件
            detailed_files = [f for f in files if f.endswith('_news_detailed.csv')]
            sentiment_files = [f for f in files if f.endswith('_daily_sentiment.csv')]
            combined_files = [f for f in files if 'combined_daily_sentiment.csv' in f]
            ticker_files = [f for f in files if f.endswith('_tickers.txt')]
            
            print(f"  📁 数据目录: {abs_path}")
            print(f"  📊 数据集股票总数: {expected_count} 只")
            print(f"  📰 详细新闻文件: {len(detailed_files)} 个")
            print(f"  📈 日度情绪文件: {len(sentiment_files)} 个")
            print(f"  📋 合并数据文件: {len(combined_files)} 个")
            print(f"  📝 股票列表文件: {len(ticker_files)} 个")
            
            if combined_files:
                collected_count = len(detailed_files)
                total_collected += collected_count
                success_rate = (collected_count / expected_count) * 100 if expected_count > 0 else 0
                print(f"  ✅ 状态: 已收集 {collected_count}/{expected_count} 只股票 ({success_rate:.1f}%)")
            else:
                print(f"  ⚠️ 状态: 数据不完整")
        else:
            print(f"  📁 数据目录: {abs_path}")
            print(f"  📊 数据集股票总数: {expected_count} 只")
            print(f"  ❌ 状态: 未收集数据")
    
    print(f"\n🎯 总体收集进度:")
    print(f"  - 数据集总股票数: {total_expected} 只")
    print(f"  - 已收集股票数: {total_collected} 只")
    overall_rate = (total_collected / total_expected) * 100 if total_expected > 0 else 0
    print(f"  - 整体完成率: {overall_rate:.1f}%")

def main():
    """主函数"""
    print("🎉 欢迎使用新闻数据收集系统！(升级版API - 高速模式)")
    print("⚡ 特性：无限制调用 + 大批次处理 + 快速收集")
    
    # 检查依赖
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"⚠️ 缺少以下依赖包: {', '.join(missing_deps)}")
        print("建议先运行选项 6 安装依赖包")
    
    while True:
        show_menu()
        choice = input("\n请选择操作 (0-6): ").strip()
        
        if choice == '0':
            print("👋 感谢使用，再见！")
            break
            
        elif choice == '1':
            print("📱 开始收集 NASDAQ 新闻数据 (高速模式)...")
            run_script('prepare_nasdaq_news_data.py', 'NASDAQ')
            
        elif choice == '2':
            print("📊 开始收集 S&P 500 新闻数据 (高速模式)...")
            run_script('prepare_sp500_news_data.py', 'S&P 500')
            
        elif choice == '3':
            print("🏛️ 开始收集 NYSE 新闻数据 (高速模式)...")
            run_script('prepare_nyse_news_data.py', 'NYSE')
            
        elif choice == '4':
            print("🌍 开始收集所有市场新闻数据 (升级版API高速模式)...")
            
            markets = [
                ('prepare_nasdaq_news_data.py', 'NASDAQ'),
                ('prepare_sp500_news_data.py', 'S&P 500'),
                ('prepare_nyse_news_data.py', 'NYSE')
            ]
            
            start_time = datetime.now()
            successful_collections = 0
            
            for script, market in markets:
                print(f"\n⏳ 等待2秒后开始收集 {market} 数据...")  # 从5秒减少到2秒
                time.sleep(2)
                
                if run_script(script, market):
                    successful_collections += 1
                    print(f"✅ {market} 收集完成")
                else:
                    print(f"❌ {market} 收集失败")
                
                # 市场间休息 - 大幅减少等待时间
                if market != 'NYSE':  # 不是最后一个
                    print(f"💤 休息5秒后继续下一个市场...")  # 从30秒减少到5秒
                    time.sleep(5)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n🎯 所有市场收集完成！(升级版API高速模式)")
            print(f"📊 总结:")
            print(f"  - 成功收集: {successful_collections}/{len(markets)} 个市场")
            print(f"  - 总用时: {duration}")
            print(f"  - 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  - 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  - 收集效率: 升级版API高速优化")
            
        elif choice == '5':
            check_data_status()
            
        elif choice == '6':
            install_dependencies()
            
        else:
            print("❌ 无效选择，请重新输入")
        
        # 暂停一下让用户看到结果
        input("\n按回车键继续...")

if __name__ == '__main__':
    main() 