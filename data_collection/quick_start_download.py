import subprocess
import sys
from pathlib import Path

def main():
    script_path = Path(__file__).parent / "download_all_data.py"
    
    print("="*70)
    print("  欢迎使用 Polygon.io 全量历史数据下载向导 (Pro 版本)")
    print("="*70)
    print("\n该脚本将执行以下操作:")
    print("1. 自动从 Polygon.io API 获取所有在主要交易所 (NYSE, NASDAQ等) 上市的股票代码。")
    print("2. 为获取到的每一只股票，下载从 2000-01-01 到现在的全部日线历史数据。")
    print("3. 利用您的高级API密钥，以极高的并发速度进行下载，最大化下载效率。")
    print("4. 将所有数据分类保存在 `../dataset_full` 文件夹下，按交易所分子目录，")
    print("   每只股票一个CSV文件。例如: `../dataset_full/XNYS/IBM.csv`")
    print("\n⚠️  警告:")
    print("- 这将下载数千个文件，可能占用数 GB 的磁盘空间。请确保您有足够的存储空间。")
    print("- 整个过程可能需要较长时间，具体取决于您的网络和系统性能。")
    print("- 请确保您的网络连接稳定。")
    print(f"\n脚本路径: {script_path}")
    print("="*70)

    try:
        choice = input("\n您是否要开始下载？(y/n): ").strip().lower()
        if choice == 'y':
            print("\n好的，即将开始下载... 您可以随时按 Ctrl+C 停止。")
            process = subprocess.Popen([sys.executable, str(script_path)], stdout=sys.stdout, stderr=sys.stderr)
            process.wait()
        else:
            print("\n操作已取消。")
    
    except KeyboardInterrupt:
        print("\n操作被用户中断。")
    except Exception as e:
        print(f"\n发生错误: {e}")

if __name__ == "__main__":
    main() 