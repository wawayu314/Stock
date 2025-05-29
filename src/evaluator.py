import numpy as np
import pandas as pd


def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match' # 断言：真实值和预测值的形状必须匹配
    performance = {} # 初始化性能字典
    # mse (Mean Squared Error, 均方误差)
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask) ** 2 / np.sum(mask)
    # IC (Information Coefficient, 信息系数)
    df_pred = pd.DataFrame(prediction * mask) # 将预测值乘以掩码后转换为DataFrame
    df_gt = pd.DataFrame(ground_truth * mask) # 将真实值乘以掩码后转换为DataFrame

    ic = [] # 存储每日IC值
    mrr_top = 0.0 # Mean Reciprocal Rank for top 1 prediction (未使用或已废弃)
    all_miss_days_top = 0 # Days when top 1 prediction missed (未使用或已废弃)
    bt_long = 1.0 # Backtesting long strategy (未使用或已废弃)
    bt_long5 = 1.0 # Backtesting top 5 long strategy (未使用或已废弃)
    bt_long10 = 1.0 # Backtesting top 10 long strategy (未使用或已废弃)
    irr = 0.0 # Information Ratio related calculation (未使用或已废弃)
    sharpe_li5 = [] # 存储每日top 5策略的收益率，用于计算夏普比率
    prec_10 = [] # 存储每日top 10预测的准确率 (收益率为正的比例)

    # 按天（列）遍历预测期
    for i in range(prediction.shape[1]):
        # IC: 计算当日预测值和真实值的相关系数
        ic.append(df_pred[i].corr(df_gt[i]))

        # 对当天的真实收益率进行排序，得到排名
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = set() # 真实收益率最高的1只股票
        gt_top5 = set() # 真实收益率最高的5只股票
        gt_top10 = set() # 真实收益率最高的10只股票

        # 从高到低遍历真实收益率排名，填充top N集合，跳过被掩码的股票
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j] # 当前排名对应的股票索引
            if mask[cur_rank][i] < 0.5: # 如果该股票当日被掩码，则跳过
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)

        # 对当天的预测收益率进行排序，得到排名
        rank_pre = np.argsort(prediction[:, i])
        pre_top1 = set() # 预测收益率最高的1只股票
        pre_top5 = set() # 预测收益率最高的5只股票
        pre_top10 = set() # 预测收益率最高的10只股票
        # 从高到低遍历预测收益率排名，填充top N集合，跳过被掩码的股票
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)

        top1_pos_in_gt = 0 # (未使用或已废弃的 MRR 相关计算)
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt

        # 获取预测收益率最高的股票的真实收益率 (未使用或已废弃的 bt_long 相关计算)
        # real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        # bt_long += real_ret_rat_top
        # gt_irr = 0.0 (未使用或已废弃)

        # for gt in gt_top10: (未使用或已废弃)
        #     gt_irr += ground_truth[gt][i]

        real_ret_rat_top5 = 0 # 当日预测top 5股票的平均真实收益率
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        # irr += real_ret_rat_top5 (未使用或已废弃)
        if len(pre_top5) > 0 : # 避免除以零
             real_ret_rat_top5 /= len(pre_top5)
        else:
             real_ret_rat_top5 = 0 # 如果没有有效的top5预测，则收益率为0
        # bt_long5 += real_ret_rat_top5 (未使用或已废弃)

        prec = 0.0 # 当日预测top 10股票中真实收益率为正的比例
        real_ret_rat_top10 = 0 # 当日预测top 10股票的平均真实收益率
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
            prec += (ground_truth[pre][i] >= 0) # 如果真实收益率大于等于0，则计数
        
        if len(pre_top10) > 0: # 避免除以零
            prec_10.append(prec / len(pre_top10))
            real_ret_rat_top10 /= len(pre_top10)
        else:
            prec_10.append(0.0) # 如果没有有效的top10预测，则准确率为0
            real_ret_rat_top10 = 0
            
        # bt_long10 += real_ret_rat_top10 (未使用或已废弃)
        sharpe_li5.append(real_ret_rat_top5) # 添加当日top 5平均收益率用于后续夏普比率计算

    performance['IC'] = np.mean(ic) if len(ic) > 0 else 0 # 计算平均IC
    performance['RIC'] = (np.mean(ic) / np.std(ic)) if len(ic) > 1 and np.std(ic) !=0 else 0 # 计算Rank IC (IC的均值/IC的标准差)，避免除零
    sharpe_li5 = np.array(sharpe_li5)
    # 计算top 5策略的夏普比率 (年化，假设一年有 252/5 = 15.87 个5日周期，原代码为15.87，可能是基于特定交易日或调整因子)
    # 更常见的年化因子是 sqrt(252) / sqrt(持有期天数)。如果sharpe_li5是每日收益，则*sqrt(252)。这里持有期是5天，所以*sqrt(252/5)。
    # 原代码的 15.87 约等于 sqrt(252)，所以可能sharpe_li5已经是某种调整后的序列。
    # 为保持一致，暂时保留原计算方式，但需注意其含义。
    if len(sharpe_li5) > 1 and np.std(sharpe_li5) != 0:
        performance['sharpe5'] = (np.mean(sharpe_li5) / np.std(sharpe_li5)) * 15.87 
    else:
        performance['sharpe5'] = 0
        
    performance['prec_10'] = np.mean(prec_10) if len(prec_10) > 0 else 0 # 计算平均top 10准确率
    return performance



