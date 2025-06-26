# 全量数据下载模块

该模块用于从 Polygon.io 下载海量的历史股票数据，以备后续研究和补充实验使用。

## 文件说明

- `download_all_data.py`: 这是核心的下载脚本。它使用 `asyncio` 异步执行，可以利用 Pro API 密钥无速率限制的优势，以极高的并发量快速下载数据。
- `quick_start_download.py`: 这是一个用户友好的启动向导。直接运行此文件即可开始下载过程。

## 如何使用

1.  确保你已经安装了必要的库：
    ```bash
    pip install pandas aiohttp tqdm
    ```
2.  直接在终端中运行启动向导：
    ```bash
    python data_collection/quick_start_download.py
    ```
3.  根据提示确认后，脚本将自动开始下载。

## 数据结构

下载的数据将存放于 `dataset_full` 目录中，结构如下：

```
dataset_full/
├── XNAS/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── ...
├── XNYS/
│   ├── IBM.csv
│   ├── JPM.csv
│   └── ...
└── ... (其他交易所)
```

每个 `csv` 文件包含以下列: `date`, `open`, `high`, `low`, `close`, `volume`。 