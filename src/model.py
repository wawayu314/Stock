import torch
import torch.nn as nn
import torch.nn.functional as F

acv = nn.GELU()

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    # 移除了 volatility 参数和相关处理逻辑
    device = prediction.device
    # volatility = volatility.to(device) # volatility comes as (batch_size, 1)

    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    
    # 回归到标准的MSE损失
    if torch.sum(mask) > 0:
        reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask, reduction='sum') / torch.sum(mask)
    else:
        reg_loss = torch.tensor(0.0, device=device)

    # # Calculate volatility-weighted regression loss
    # epsilon = 1e-6 # 一个小常数，用于防止除零错误并稳定计算
    # # Weights: inversely proportional to volatility. Higher volatility = lower weight for its error.
    # # Adding abs() just in case, though std dev should be non-negative.
    # # Clamping volatility at a minimum value (epsilon) before division can also be good.
    # clamped_volatility = torch.clamp(volatility.abs(), min=epsilon) # 形状 (batch_size, 1)
    # loss_weights = 1.0 / clamped_volatility # 形状 (batch_size, 1)
    
    # # 新增：裁剪权重以防止极端值
    # loss_weights = torch.clamp(loss_weights, min=0.5, max=2.0) 
    
    # # 可选：归一化权重，使其均值为1，以保持损失量级相似
    # # This might not be strictly necessary but can help with hyperparameter stability.
    # # if torch.sum(mask) > 0: # 如果掩码全为零，则避免除零错误
    # #     loss_weights = loss_weights * (torch.sum(mask) / torch.sum(loss_weights * mask)) # 基于掩码元素的归一化
    # # A simpler normalization if all elements are usually valid or mask is applied later:
    # loss_weights = loss_weights / (torch.mean(loss_weights) + epsilon)

    # # Calculate squared errors per element, then apply mask and weights
    # squared_errors = (return_ratio - ground_truth)**2 # 形状 (batch_size, 1)
    
    # # Apply mask first to errors, then weights. Or apply mask to weighted errors.
    # # Weighted and masked squared errors
    # weighted_masked_squared_errors = loss_weights * squared_errors * mask # 逐元素相乘
    
    # # Sum of weighted masked squared errors divided by sum of mask (to average over valid entries)
    # if torch.sum(mask) > 0:
    #     reg_loss = torch.sum(weighted_masked_squared_errors) / torch.sum(mask)
    # else: # 如果掩码全为零 (例如，在某些边缘情况的批次中)，则避免除零错误
    #     reg_loss = torch.tensor(0.0, device=device)

    # 排序损失计算 (保持不变)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class SequentialFeatureWeightingNet(nn.Module):
    def __init__(self, num_features, conv_kernel_size, mlp_hidden_dim, dropout_rate=0.1):
        super(SequentialFeatureWeightingNet, self).__init__()
        self.num_features = num_features

        # 一维卷积层，独立处理每个特征的时间序列
        self.conv1d = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features, # 每个输入特征序列输出一个处理后的序列
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2, # 对于奇数卷积核大小使用 'same' 填充
            groups=num_features # 关键：独立处理每个通道 (特征)
        )
        self.activation_conv = nn.ReLU() # 或 nn.GELU()

        # MLP 从处理后的、池化的特征中生成权重
        # 此 MLP 的输入将是 num_features (在全局平均池化之后)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, num_features)
        )

    def forward(self, x_input_n_t_k):
        # x_input_n_t_k 形状: (N, T, K) - 批次, 时间, 特征
        
        # 置换为 (N, K, T) 以适应 Conv1D，因为 Conv1D 期望通道作为第二个维度
        x_permuted_n_k_t = x_input_n_t_k.permute(0, 2, 1) # 形状: (N, K_特征数, T_时间步长)
        
        # 独立地对每个特征的时间序列应用卷积
        x_conv_processed = self.conv1d(x_permuted_n_k_t) # 形状: (N, K_特征数, T_卷积后时间步长)
        x_activated = self.activation_conv(x_conv_processed)
        
        # 在时间维度上进行全局平均池化
        # 池化输入: (N, K_特征数, T_卷积后时间步长)
        # 输出: (N, K_特征数)
        x_pooled_n_k = torch.mean(x_activated, dim=2) 
        
        # MLP 获取特征的 logits/分数
        feature_logits = self.mlp(x_pooled_n_k) # 形状: (N, K_特征数)
        
        # 应用 sigmoid 获取每个特征的独立权重 (门控)
        weights_n_k = torch.sigmoid(feature_logits) # 形状: (N, K_特征数)
        
        # 扩展维度以允许与 (N, T, K) 进行广播
        # 输出权重形状: (N, 1, K_特征数)
        return weights_n_k.unsqueeze(1)


class AttentiveFeatureWeightingNet(nn.Module):
    def __init__(self, num_features, attention_hidden_dim, dropout_rate=0.1):
        super(AttentiveFeatureWeightingNet, self).__init__()
        self.num_features = num_features
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(num_features, attention_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(attention_hidden_dim, num_features)
        )

    def forward(self, x_input_n_t_k):
        # x_input_n_t_k 形状: (N, T, K) - 批次, 时间, 特征
        # 在时间维度上对特征进行平均
        x_avg_time = torch.mean(x_input_n_t_k, dim=1) # 形状: (N, K)

        # 获取每个特征的注意力分数/logits
        attention_logits = self.attention_mlp(x_avg_time) # 形状: (N, K)

        # 应用 sigmoid 获取每个特征的独立权重 (门控)
        weights_n_k = torch.sigmoid(attention_logits) # 形状: (N, K)

        # 扩展维度以允许与 (N, T, K) 进行广播
        # 输出权重形状: (N, 1, K)
        return weights_n_k.unsqueeze(1)


class FeatureWeightingNet(nn.Module):
    def __init__(self, num_features, hidden_dim, history_len):
        super(FeatureWeightingNet, self).__init__()
        self.num_features = num_features
        self.history_len = history_len
        # 处理时间平均特征的层
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x 形状: (batch_size, num_stocks_in_item, num_features, history_len)
        # 对于 StockMixer，num_stocks_in_item 将为 1，因为每个批次项是一个股票序列。
        # 在时间维度上对特征进行平均
        x_avg_time = torch.mean(x, dim=3) # 形状: (batch_size, num_stocks_in_item, num_features)

        # 获取每个特征的权重
        weights = self.mlp(x_avg_time) # 形状: (batch_size, num_stocks_in_item, num_features)

        # 将权重重塑为 (batch_size, num_stocks_in_item, num_features, 1) 以进行广播
        weights = weights.unsqueeze(-1)
        return weights


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0):
        super(MixerBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.LN = acv
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x


class Mixer2d(nn.Module):
    def __init__(self, time_steps, channels):
        super(Mixer2d, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = MixerBlock(time_steps, time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class TriU(nn.Module):
    def __init__(self, time_step):
        super(TriU, self).__init__()
        self.time_step = time_step
        if self.time_step < 0:
            raise ValueError("time_step 不能为负数")
        
        self.triU = nn.ParameterList(
            [
                nn.Linear(i + 1, 1)
                for i in range(self.time_step)
            ]
        )

    def forward(self, inputs):
        if self.time_step == 0:
            if inputs.ndim < 2:
                 raise ValueError("输入张量至少需要2个维度才能用于 TriU")
            output_shape = list(inputs.shape)
            output_shape[-1] = 0
            return torch.empty(output_shape, device=inputs.device, dtype=inputs.dtype)

        actual_input_time_dim = inputs.size(-1)
        
        effective_time_step = min(self.time_step, actual_input_time_dim)

        if effective_time_step == 0:
            output_shape = list(inputs.shape)
            output_shape[-1] = 0
            return torch.empty(output_shape, device=inputs.device, dtype=inputs.dtype)

        current_slice = inputs[..., 0:1]
        x = self.triU[0](current_slice) 
        
        for i in range(1, effective_time_step):
            current_slice = inputs[..., 0:i+1]
            processed_slice = self.triU[i](current_slice)
            x = torch.cat([x, processed_slice], dim=-1)
        return x


class TimeMixerBlock(nn.Module):
    def __init__(self, time_step):
        super(TimeMixerBlock, self).__init__()
        self.time_step = time_step
        self.dense_1 = TriU(time_step)
        self.LN = acv
        self.dense_2 = TriU(time_step)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        x = self.dense_2(x)
        return x


class MultiScaleTimeMixer(nn.Module): # This is the old one, might be unused. Keeping for reference if needed.
    def __init__(self, time_step, channel, scale_count=1):
        super(MultiScaleTimeMixer, self).__init__()
        self.time_step = time_step
        self.scale_count = scale_count
        self.mix_layer = nn.ParameterList([nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2 ** i, stride=2 ** i),
            TriU(int(time_step / 2 ** i)),
            nn.Hardswish(),
            TriU(int(time_step / 2 ** i))
        ) for i in range(scale_count)])
        self.mix_layer[0] = nn.Sequential(
            nn.LayerNorm([time_step, channel]),
            TriU(int(time_step)),
            nn.Hardswish(),
            TriU(int(time_step))
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.mix_layer[0](x)
        for i in range(1, self.scale_count):
            y = torch.cat((y, self.mix_layer[i](x)), dim=-1)
        return y


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps, channels):
        super(Mixer2dTriU, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    def __init__(self, time_step, channel, scale_dim=8):
        super(MultTime2dMixer, self).__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs, y):
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20):
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs):
        # 输入张量的形状应为 (..., stocks)，例如 (T_concat, K, N_stocks)
        x = self.layer_norm_stock(inputs) # 在最后一个维度 (N_stocks) 上应用 LayerNorm
        x = self.dense1(x)                # 在最后一个维度 (N_stocks -> hidden_dim) 上应用线性层
        x = self.activation(x)
        x = self.dense2(x)                # 在最后一个维度 (hidden_dim -> N_stocks) 上应用线性层
        return x


class PermuteForConv1d(nn.Module):
    """将 (批次, 时间, 通道) 张量置换为 (批次, 通道, 时间)。"""
    def forward(self, x):
        return x.permute(0, 2, 1)


class PermuteFromConv1d(nn.Module):
    """将 (批次, 通道, 时间) 张量置换回 (批次, 时间, 通道)。"""
    def forward(self, x):
        return x.permute(0, 2, 1)


class MultiScaleTimeMixerRevised_OriginalStyle(nn.Module): # 重命名以区分，并恢复原始风格
    def __init__(self, initial_time_steps, channels, scales_config, activation_fn_class=nn.Hardswish):
        super().__init__()
        self.channels = channels
        self.initial_time_steps = initial_time_steps

        self.scale_processing_modules = nn.ModuleList() # 用于存储每个尺度的处理模块
        
        _calculated_concat_time_dim = 0

        # 分支 1: 原始尺度
        self.scale_processing_modules.append(nn.Identity()) # 原始尺度直接通过
        _calculated_concat_time_dim += initial_time_steps

        # 卷积尺度
        # scales_config 现在只包含 {'kernel': K, 'stride': S}
        for conv_config in scales_config:
            kernel_size = conv_config['kernel']
            stride = conv_config['stride']
            
            # 动态计算此卷积尺度输出的时间维度 (假设 padding=0)
            # T_out = floor((T_in - K)/S) + 1
            current_scale_output_time_dim = (initial_time_steps - kernel_size) // stride + 1

            if current_scale_output_time_dim <= 0:
                print(f"警告: K={kernel_size}, S={stride} 的卷积尺度导致输出时间维度非正。跳过此尺度。")
                continue
            
            # 构建此卷积尺度的处理序列
            self.scale_processing_modules.append(
                nn.Sequential(
                    PermuteForConv1d(), # (B,T_initial,C) -> (B,C,T_initial)
                    nn.Conv1d(in_channels=channels, out_channels=channels,
                              kernel_size=kernel_size, stride=stride, padding=0), # 明确 padding=0
                    PermuteFromConv1d(), # (B,C,T_current_scale) -> (B,T_current_scale,C)
                    nn.LayerNorm([current_scale_output_time_dim, channels]) # 在 (T_current_scale, C) 上进行LN
                )
            )
            _calculated_concat_time_dim += current_scale_output_time_dim
        
        if _calculated_concat_time_dim == 0: # 如果只有原始尺度失败了（不太可能）或所有卷积尺度都失败了
             raise ValueError("MultiScaleTimeMixerRevised_OriginalStyle: _calculated_concat_time_dim 为 0。没有有效的尺度。")

        # 用于处理拼接后特征的共享TriU网络
        # 注意：TriU作用于最后一个维度，所以输入TriU网络的张量应该是 (B, C, T_concat)
        self.shared_triu_processor = nn.Sequential(
            PermuteForConv1d(), # (B, T_concat, C) -> (B, C, T_concat)
            TriU(_calculated_concat_time_dim),
            activation_fn_class(),
            TriU(_calculated_concat_time_dim),
            PermuteFromConv1d()  # (B, C, T_concat_processed) -> (B, T_concat_processed, C)
        )
        self.final_concat_time_dim = _calculated_concat_time_dim # 存储计算得到的总时间维度

    def forward(self, inputs):
        # inputs: (batch_size, initial_time_steps, channels)
        
        scale_outputs = []
        
        # 应用每个尺度处理模块
        # 第一个模块 (nn.Identity) 处理原始尺度
        scale_outputs.append(self.scale_processing_modules[0](inputs)) 
        
        # 后续模块处理卷积尺度，它们都以原始的 'inputs' 作为输入
        for i in range(1, len(self.scale_processing_modules)):
            scale_outputs.append(self.scale_processing_modules[i](inputs))

        # 沿时间维度 (dim=1) 拼接所有尺度的输出
        # 每个 scale_output 的形状是 (B, T_scale_i, C)
        concatenated_features = torch.cat(scale_outputs, dim=1) # 形状: (B, final_concat_time_dim, C)
        
        # 通过共享的TriU处理器进行处理
        output = self.shared_triu_processor(concatenated_features) # 输出形状: (B, final_concat_time_dim, C)
        return output


class StockMixer(nn.Module):
    def __init__(self, stocks, time_steps, channels, market, scale, # scale 参数可能不再直接使用或含义改变
                 feature_weighting_hidden_dim=None, 
                 feature_weighting_conv_kernel_size=3,
                 activation_fn_class=nn.Hardswish):
        """
        stocks: 股票数量 (例如，来自 train.py 的 stock_num)。
        time_steps: 初始回溯长度 (例如，来自 train.py 的 lookback_length)。
        channels: 每个时间步长每个股票的输入特征数量 (例如，来自 train.py 的 fea_num)。
        market: 用于区分不同市场的处理逻辑 (例如，'sp500' vs 其他)。
        scale: (原始含义可能已废弃) 在SP500中，实际尺度配置由内部定义。
        feature_weighting_hidden_dim: 特征加权网络 MLP部分的隐藏维度。
        feature_weighting_conv_kernel_size: SequentialFeatureWeightingNet 中 Conv1D 的卷积核大小。
        activation_fn_class: 要使用的激活函数的类 (例如，nn.Hardswish, nn.GELU)。
        """
        super(StockMixer, self).__init__()
        self.time_steps = time_steps
        self.stocks = stocks
        self.channels = channels
        # self.scale = scale # scale 参数可能不再像以前那样使用
        self.market = market.lower()
        self.activation_fn = activation_fn_class()

        self.feature_weighting_net = None
        if feature_weighting_hidden_dim is not None and feature_weighting_hidden_dim > 0 and feature_weighting_conv_kernel_size > 0:
            self.feature_weighting_net = SequentialFeatureWeightingNet(
                num_features=channels, 
                conv_kernel_size=feature_weighting_conv_kernel_size,
                mlp_hidden_dim=feature_weighting_hidden_dim,
                dropout_rate=0.1 
            )

        self.mlp1 = nn.Linear(self.time_steps, self.time_steps)
        self.activation1 = self.activation_fn
        self.mlp2 = nn.Linear(self.time_steps, self.time_steps)
        self.activation2 = self.activation_fn
        self.mlp3 = nn.Linear(self.channels, self.channels)
        self.activation3 = self.activation_fn
        self.mlp4 = nn.Linear(self.channels, self.channels)
        self.activation4 = self.activation_fn

        if self.market == 'sp500':
            # 恢复到最初提议的 scales_config 结构，只包含 kernel 和 stride
            scales_config_sp500_original_style = [
                {'kernel': 2, 'stride': 2}, 
                {'kernel': 3, 'stride': 1},
                # 可以根据需要添加更多尺度，例如：
                # {'kernel': 5, 'stride': 1}, 
            ]
            self.time_mixer = MultiScaleTimeMixerRevised_OriginalStyle( # 使用新恢复的类
                initial_time_steps=self.time_steps,
                channels=self.channels,
                scales_config=scales_config_sp500_original_style, 
                activation_fn_class=activation_fn_class
            )
            self.concat_time_dim = self.time_mixer.final_concat_time_dim # 从模块获取拼接后的时间维度
        else: # 其他市场 (如 NASDAQ)
            self.time_mixer = TimeMixerBlock(self.time_steps)
            self.concat_time_dim = self.time_steps 

        # NoGraphMixer 的 hidden_dim。原始论文中这个部分可能比较简单，或者有特定设置。
        # 保持之前的逻辑，即 hidden_dim = self.concat_time_dim * self.channels，这似乎是一个非常大的值。
        # 或者，使用一个更常规的值，例如 self.stocks // 2 或固定的较小值如20或64。
        # 为了尽可能恢复，我们暂时保留原始的计算方式，尽管它可能不是最优的。
        # 一个常见的做法是让 hidden_dim 成为 stocks 的一个比例或者一个固定的中等大小的数。
        # 如果 hidden_dim=self.concat_time_dim * self.channels 导致过大或不合理，可以调整为例如：
        # nogm_hidden_dim = max(20, self.stocks // 4) 
        nogm_hidden_dim = self.concat_time_dim * self.channels # 沿用之前的设置，但需要注意其合理性
        if nogm_hidden_dim == 0 and self.stocks > 0 : # 避免hidden_dim为0，至少为1
             nogm_hidden_dim = self.stocks 
        elif self.stocks == 0: # 如果 stocks 为 0，则 hidden_dim 也应为0或引发错误
             nogm_hidden_dim = 0


        self.stock_mixer = NoGraphMixer(stocks=self.stocks, hidden_dim=nogm_hidden_dim if nogm_hidden_dim > 0 else self.stocks if self.stocks > 0 else 1)
        
        self.time_reduction_fc = nn.Linear(self.concat_time_dim, 1) 

        if self.channels > 1:
            self.final_channel_reducer = nn.Linear(self.channels, 1) 
        # 如果 self.channels 为 1，则不需要此层。

    def forward(self, inputs):
        # inputs: (batch_size_is_stock_num, time_steps, channels) -> (N, T, K)
        current_x = inputs
        if self.feature_weighting_net:
            weights_n_1_k = self.feature_weighting_net(current_x) 
            current_x = current_x * weights_n_1_k

        x_time = current_x.permute(0, 2, 1)  # N, K, T
        x_time = self.mlp1(x_time)
        x_time = self.activation1(x_time)
        x_time = self.mlp2(x_time)
        x_time = self.activation2(x_time)  # N, K, T

        x_channel = x_time.permute(0, 2, 1)  # N, T, K
        x_channel = self.mlp3(x_channel)
        x_channel = self.activation3(x_channel)
        x_channel = self.mlp4(x_channel)
        x_channel = self.activation4(x_channel)  # N, T, K

        # 应用时间混合器
        # MultiScaleTimeMixerRevised_OriginalStyle 和 TimeMixerBlock 都期望 (N, T, K) 或 (N, K, T)
        # 并产生 (N, T_processed, K) 或 (N, K, T_processed)
        
        if isinstance(self.time_mixer, MultiScaleTimeMixerRevised_OriginalStyle):
            # MultiScaleTimeMixerRevised_OriginalStyle.forward 期望 (N, T_initial, K)
            # 并输出 (N, T_concat, K)
            y_temporal_mixed = self.time_mixer(x_channel) # x_channel is (N, T, K)
        elif isinstance(self.time_mixer, TimeMixerBlock):
            # TimeMixerBlock 期望 (..., T_to_process)
            x_permuted_for_tmb = x_channel.permute(0, 2, 1)  # (N, K, T)
            y_processed_on_time_dim = self.time_mixer(x_permuted_for_tmb) # (N, K, T)
            y_temporal_mixed = y_processed_on_time_dim.permute(0, 2, 1) # (N, T, K)
        else:
            raise ValueError(f"不支持的时间混合器类型: {type(self.time_mixer)}")

        # y_temporal_mixed 的形状是 (N, self.concat_time_dim, K)

        y_for_stock_mixer = y_temporal_mixed.permute(1, 2, 0) # (T_concat, K, N)
        y_stock_mixed = self.stock_mixer(y_for_stock_mixer)   # (T_concat, K, N)
        y_stock_mixed = y_stock_mixed.permute(2, 0, 1)        # (N, T_concat, K)

        y_time_input = y_stock_mixed.permute(0, 2, 1)               # (N, K, T_concat)
        y_time_reduced = self.time_reduction_fc(y_time_input)       # (N, K, 1)
        
        y_channel_input = y_time_reduced.squeeze(-1) # (N, K)
        
        if self.channels == 1:
            final_pred = y_channel_input 
            if final_pred.ndim == 1: 
                final_pred = final_pred.unsqueeze(-1)
        else:
            if hasattr(self, 'final_channel_reducer'):
                final_pred = self.final_channel_reducer(y_channel_input) # (N, 1)
            else:
                final_pred = y_channel_input[:, 0:1] # Fallback
            
        return final_pred

