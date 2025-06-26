# -*- coding: utf-8 -*-
"""
S&P 500股票行业分类数据
用于实现基于行业的稀疏注意力机制
"""

# S&P 500主要行业分类（简化版11个GICS行业）
SP500_SECTOR_CLASSIFICATION = {
    # 信息技术 Information Technology
    'Information Technology': [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 
        'AVGO', 'ORCL', 'CRM', 'ACN', 'ADBE', 'CSCO', 'AMD', 'INTC', 
        'TXN', 'QCOM', 'INTU', 'IBM', 'AMAT', 'ADI', 'MU', 'LRCX', 
        'KLAC', 'MRVL', 'FTNT', 'CDNS', 'SNPS', 'ADSK', 'ROP', 'CTSH', 
        'FISV', 'MSI', 'APH', 'FICO', 'PAYX', 'ANSS', 'MPWR', 'TYL', 
        'GDDY', 'STX', 'WDC', 'HPQ', 'NTAP', 'IT', 'TER', 'SWKS', 
        'MCHP', 'CDW', 'ANET', 'KEYS', 'GLW', 'HPE', 'EPAM', 'JNPR'
    ],
    
    # 医疗保健 Health Care
    'Health Care': [
        'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'LLY', 'ABT', 'DHR', 
        'BMY', 'CVS', 'MDT', 'GILD', 'CI', 'ISRG', 'REGN', 'VRTX', 'ELV', 
        'ZTS', 'DXCM', 'BSX', 'HUM', 'SYK', 'AMGN', 'BDX', 'EW', 'A', 
        'ANTM', 'CNC', 'MRNA', 'IQV', 'RMD', 'IDXX', 'ILMN', 'BIIB', 
        'HCA', 'COO', 'WAT', 'MTD', 'ALGN', 'TECH', 'BIO', 'XRAY', 
        'MOH', 'DGX', 'RVTY', 'CRL', 'CAH', 'MCK', 'ABC', 'UHS'
    ],
    
    # 金融服务 Financials
    'Financials': [
        'BRK.B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 
        'CB', 'PGR', 'MMC', 'ICE', 'CME', 'PNC', 'AON', 'USB', 'TFC', 'AJG', 
        'COF', 'AIG', 'MET', 'PRU', 'ALL', 'TRV', 'DFS', 'AFL', 'AMP', 'FIS', 
        'BK', 'STT', 'TROW', 'SCHW', 'MTB', 'FITB', 'HBAN', 'RF', 'CFG', 
        'KEY', 'WRB', 'L', 'PFG', 'FDS', 'MSCI', 'MKTX', 'CBOE', 'RJF', 
        'NDAQ', 'IVZ', 'NTB', 'PBCT', 'CMA', 'ZION', 'SIVB', 'NTRS'
    ],
    
    # 非必需消费品 Consumer Discretionary
    'Consumer Discretionary': [
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 
        'CMG', 'ORLY', 'AZO', 'LULU', 'YUM', 'GM', 'F', 'MAR', 'HLT', 
        'ABNB', 'RCL', 'CCL', 'NCLH', 'MGM', 'LVS', 'WYNN', 'DRI', 'QSR', 
        'SBAC', 'EBAY', 'ETSY', 'DECK', 'TPG', 'KMX', 'BBY', 'DG', 'DLTR', 
        'ROST', 'TGT', 'COST', 'WMT', 'KR', 'SYY', 'TSCO', 'DHI', 'LEN', 
        'PHM', 'NVR', 'KBH', 'TOL', 'WHR', 'TPX', 'MHK', 'CARR'
    ],
    
    # 工业 Industrials
    'Industrials': [
        'BA', 'CAT', 'UPS', 'RTX', 'HON', 'UNP', 'LMT', 'DE', 'MMM', 'GE', 
        'FDX', 'WM', 'EMR', 'ETN', 'ITW', 'CSX', 'NSC', 'AON', 'PH', 'CMI', 
        'GD', 'NOC', 'TDG', 'PCAR', 'RSG', 'URI', 'GWW', 'FAST', 'PAYX', 
        'ROK', 'OTIS', 'J', 'CHRW', 'PWR', 'EXPD', 'VRSK', 'XYL', 'IEX', 
        'DOV', 'FTV', 'BR', 'LUV', 'DAL', 'UAL', 'AAL', 'JBHT', 'ODFL', 
        'LDOS', 'HII', 'TXT', 'ROL', 'ALLE', 'MAS', 'AOS', 'GNRC'
    ],
    
    # 必需消费品 Consumer Staples  
    'Consumer Staples': [
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'MDLZ', 'CL', 'GIS', 'KMB', 'SYY', 
        'ADM', 'MNST', 'CHD', 'K', 'CPB', 'HRL', 'TSN', 'CAG', 'MKC', 'SJM', 
        'LW', 'TAP', 'BF.B', 'STZ', 'DG', 'DLTR', 'KR', 'SYY', 'EXR', 'PSA', 
        'WBA', 'CVS', 'PM', 'MO', 'EL', 'CLX', 'CTAS', 'KMX', 'TGT', 'ROST', 
        'SBUX', 'YUM', 'MCD', 'CMG', 'QSR', 'DRI', 'TXRH', 'PNRA', 'JACK'
    ],
    
    # 能源 Energy
    'Energy': [
        'XOM', 'CVX', 'COP', 'EPD', 'KMI', 'SLB', 'PSX', 'MPC', 'VLO', 'WMB', 
        'OKE', 'PXD', 'EOG', 'BKR', 'HAL', 'DVN', 'FANG', 'MRO', 'OXY', 'HES', 
        'CTRA', 'APA', 'EQT', 'CHK', 'MTDR', 'SM', 'RRC', 'CNX', 'AR', 'CLR', 
        'NOV', 'FTI', 'HP', 'RIG', 'VAL', 'TRGP', 'ET', 'ENB', 'TC', 'TRP'
    ],
    
    # 公用事业 Utilities
    'Utilities': [
        'NEE', 'SO', 'DUK', 'AEP', 'SRE', 'D', 'EXC', 'XEL', 'PEG', 'WEC', 
        'ED', 'FE', 'ETR', 'ES', 'AWK', 'DTE', 'PPL', 'AEE', 'CMS', 'CNP', 
        'NI', 'LNT', 'EVRG', 'ATO', 'NRG', 'CEG', 'PCG', 'EIX', 'TSN', 'AES'
    ],
    
    # 房地产 Real Estate
    'Real Estate': [
        'AMT', 'PLD', 'CCI', 'EQIX', 'WELL', 'SPG', 'PSA', 'O', 'EXR', 'AVB', 
        'EQR', 'SBAC', 'DLR', 'VTR', 'ESS', 'MAA', 'KIM', 'UDR', 'CPT', 'FRT', 
        'REG', 'BXP', 'HST', 'PEAK', 'AIV', 'ELS', 'LSI', 'CUBE', 'AMH', 'SUI'
    ],
    
    # 材料 Materials
    'Materials': [
        'LIN', 'SHW', 'APD', 'ECL', 'FCX', 'NEM', 'DOW', 'DD', 'VMC', 'MLM', 
        'NUE', 'PPG', 'IFF', 'EMN', 'PKG', 'BALL', 'AMCR', 'AVY', 'CF', 'LYB', 
        'WRK', 'IP', 'FMC', 'CE', 'ALB', 'MOS', 'RPM', 'SEE', 'CCK', 'CRH'
    ],
    
    # 通信服务 Communication Services
    'Communication Services': [
        'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 
        'TMUS', 'NWSA', 'NWS', 'FOXA', 'FOX', 'PARA', 'WBD', 'IPG', 'OMC', 
        'TTWO', 'EA', 'ATVI', 'LYV', 'SIRI', 'PINS', 'SNAP', 'TWTR', 'DISH'
    ]
}

def get_stock_sector(ticker):
    """
    根据股票代码获取所属行业
    
    Args:
        ticker: 股票代码
        
    Returns:
        行业名称，如果找不到返回'Unknown'
    """
    for sector, stocks in SP500_SECTOR_CLASSIFICATION.items():
        if ticker in stocks:
            return sector
    return 'Unknown'

def create_sector_mapping(stock_list):
    """
    为股票列表创建行业映射
    
    Args:
        stock_list: 股票代码列表
        
    Returns:
        dict: {stock_ticker: sector_id}
    """
    sector_names = list(SP500_SECTOR_CLASSIFICATION.keys())
    sector_to_id = {name: idx for idx, name in enumerate(sector_names)}
    
    stock_to_sector = {}
    for stock in stock_list:
        sector_name = get_stock_sector(stock)
        if sector_name == 'Unknown':
            # 如果找不到对应行业，分配到最后一个类别
            sector_id = len(sector_names)
        else:
            sector_id = sector_to_id[sector_name]
        stock_to_sector[stock] = sector_id
    
    return stock_to_sector

def create_industry_attention_mask(stock_list, same_industry_only=True):
    """
    创建基于行业的注意力掩码矩阵
    
    Args:
        stock_list: 股票代码列表
        same_industry_only: 是否只允许同行业间的注意力
        
    Returns:
        torch.Tensor: 注意力掩码矩阵 (num_stocks, num_stocks)
        1表示允许注意力，0表示屏蔽注意力
    """
    import torch
    
    stock_sectors = create_sector_mapping(stock_list)
    num_stocks = len(stock_list)
    
    # 创建掩码矩阵
    mask = torch.zeros(num_stocks, num_stocks)
    
    for i, stock_i in enumerate(stock_list):
        sector_i = stock_sectors[stock_i]
        for j, stock_j in enumerate(stock_list):
            sector_j = stock_sectors[stock_j]
            
            if same_industry_only:
                # 只允许同行业股票间的注意力
                if sector_i == sector_j:
                    mask[i, j] = 1.0
            else:
                # 允许所有注意力，但给同行业更高权重
                if sector_i == sector_j:
                    mask[i, j] = 1.0
                else:
                    mask[i, j] = 0.1  # 跨行业注意力衰减
    
    return mask

def print_sector_summary(stock_list):
    """打印行业分布统计"""
    stock_sectors = create_sector_mapping(stock_list)
    sector_count = {}
    
    for stock, sector_id in stock_sectors.items():
        sector_names = list(SP500_SECTOR_CLASSIFICATION.keys())
        if sector_id < len(sector_names):
            sector_name = sector_names[sector_id]
        else:
            sector_name = 'Unknown'
            
        sector_count[sector_name] = sector_count.get(sector_name, 0) + 1
    
    print("📊 S&P 500行业分布统计:")
    print("=" * 50)
    for sector, count in sorted(sector_count.items(), key=lambda x: x[1], reverse=True):
        print(f"{sector:25s}: {count:3d} 只股票")
    
    total_stocks = len(stock_list)
    known_stocks = sum(count for sector, count in sector_count.items() if sector != 'Unknown')
    print(f"\n总计: {total_stocks} 只股票")
    print(f"已分类: {known_stocks} 只股票 ({known_stocks/total_stocks*100:.1f}%)")
    
    if 'Unknown' in sector_count:
        print(f"未分类: {sector_count['Unknown']} 只股票 ({sector_count['Unknown']/total_stocks*100:.1f}%)")

if __name__ == "__main__":
    # 测试代码
    test_stocks = ['AAPL', 'GOOGL', 'JNJ', 'JPM', 'XOM', 'TEST_UNKNOWN']
    print_sector_summary(test_stocks) 