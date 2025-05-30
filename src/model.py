import torch
import torch.nn as nn
import torch.nn.functional as F
import math

acv = nn.GELU()

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    if prediction.ndim == 1:
        prediction_col = prediction.unsqueeze(1)
    elif prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction_col = prediction
    else:
        print(f"Warning: Unexpected prediction shape in get_loss: {prediction.shape}")
        prediction_col = prediction.squeeze().unsqueeze(1) if prediction.numel() == batch_size else prediction
    return_ratio = torch.div(torch.sub(prediction_col, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    if ground_truth.ndim == 1:
        ground_truth_col = ground_truth.unsqueeze(1)
    elif ground_truth.ndim == 2 and ground_truth.shape[1] == 1:
        ground_truth_col = ground_truth
    else:
        print(f"Warning: Unexpected ground_truth shape in get_loss: {ground_truth.shape}")
        ground_truth_col = ground_truth.squeeze().unsqueeze(1) if ground_truth.numel() == batch_size else ground_truth
    gt_pw_dif = torch.sub(
        all_one @ ground_truth_col.t(),
        ground_truth_col @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


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
            raise ValueError("time_step cannot be negative")
        
        self.triU = nn.ParameterList(
            [
                nn.Linear(i + 1, 1)
                for i in range(self.time_step)
            ]
        )

    def forward(self, inputs):
        # inputs shape: (batch, channels, time_actual) or (batch, time_actual, features)
        # TriU is designed to work on the last dimension.
        
        if self.time_step == 0:
            if inputs.ndim < 2:
                 raise ValueError("Input tensor must have at least 2 dimensions for TriU")
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


class MultiScaleTimeMixer(nn.Module):
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
        x = inputs
        x = x.permute(1, 0)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x.permute(1, 0)
        return x


class PermuteForConv1d(nn.Module):
    """Permutes a (batch, time, channel) tensor to (batch, channel, time)."""
    def forward(self, x):
        return x.permute(0, 2, 1)


class PermuteFromConv1d(nn.Module):
    """Permutes a (batch, channel, time) tensor back to (batch, time, channel)."""
    def forward(self, x):
        return x.permute(0, 2, 1)


class PermuteAndApplyTriUNet(nn.Module):
    """Permutes to (B, C, T), applies a TriU-based net, and permutes back to (B, T, C)."""
    def __init__(self, triu_net_module):
        super().__init__()
        self.triu_net_module = triu_net_module

    def forward(self, x):
        # x is expected as (Batch, Time, Channels)
        # TriU (and triu_net_module) expects (Batch, Channels, Time_to_process)
        x_permuted = x.permute(0, 2, 1)  # (B, C, T)
        processed_x = self.triu_net_module(x_permuted)  # Output (B, C, T_processed)
        return processed_x.permute(0, 2, 1)  # (B, T_processed, C)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        num_cos_columns = pe[:, 1::2].shape[1]
        if num_cos_columns > 0:
            pe[:, 1::2] = torch.cos(position * div_term[:num_cos_columns])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiScaleTimeMixerRevised(nn.Module):
    def __init__(self, initial_time_steps, channels, scales_config, activation_fn_class=nn.Hardswish):
        super().__init__()
        self.channels = channels
        self.initial_time_steps = initial_time_steps

        self.conv_layers = nn.ModuleList()
        self.conv_scale_lns = nn.ModuleList() # LN for each convolutional scale output before concat
        
        concat_time_dim_for_triu = 0

        # Branch 1: Original scale contribution to concat_time_dim
        concat_time_dim_for_triu += initial_time_steps

        # Convolutional scales
        for scale_info in scales_config:
            if scale_info['type'] == 'conv':
                ts_after_conv = scale_info['time_steps']
                if ts_after_conv <= 0:
                    continue
                
                self.conv_layers.append(
                    nn.Conv1d(in_channels=channels, out_channels=channels,
                              kernel_size=scale_info['kernel_size'], stride=scale_info['stride'])
                )
                self.conv_scale_lns.append(nn.LayerNorm([ts_after_conv, channels]))
                concat_time_dim_for_triu += ts_after_conv
            elif scale_info['type'] == 'original':
                # This part of config is mainly for calculating concat_time_dim_for_triu
                # and for clarity. Original scale features are handled directly from 'inputs'.
                pass 

        if concat_time_dim_for_triu == 0:
            raise ValueError("MultiScaleTimeMixerRevised: concat_time_dim_for_triu is 0. No valid scales.")

        triu_network = nn.Sequential(
            TriU(concat_time_dim_for_triu),
            activation_fn_class(),
            TriU(concat_time_dim_for_triu)
        )
        
        # REMOVED: self.final_ln_before_permute (LayerNorm after concat was removed in the best version)
        self.shared_permute_and_triu = PermuteAndApplyTriUNet(triu_network)

    def forward(self, inputs):
        # inputs: (batch_size, initial_time_steps, channels)
        
        extracted_scale_features = []

        # Original scale feature (no LN before concat)
        original_scale_feat = inputs 
        extracted_scale_features.append(original_scale_feat)

        # Convolutional scale feature extraction
        conv_input = inputs.permute(0, 2, 1) # (B, C, initial_T)
        for i, conv_layer in enumerate(self.conv_layers):
            convolved_feat = conv_layer(conv_input) # (B, C, T_conv_scale)
            convolved_feat_permuted = convolved_feat.permute(0, 2, 1) # (B, T_conv_scale, C)
            
            # Apply LayerNorm specific to this convolutional scale
            normed_convolved_feat = self.conv_scale_lns[i](convolved_feat_permuted)
            extracted_scale_features.append(normed_convolved_feat)

        if not extracted_scale_features: # Should be at least original_scale_feat
            raise RuntimeError("No features extracted. Check scales_config and model setup.")
        
        concatenated_features = torch.cat(extracted_scale_features, dim=1)
        
        # Concatenated features directly go to the TriU network (via PermuteAndApplyTriUNet)
        output = self.shared_permute_and_triu(concatenated_features)
        
        return output


# --- GCN Layer and Adjacency Normalization --- 
def normalize_adjacency_matrix(adj, self_loops=True):
    """对邻接矩阵进行归一化处理: D^-0.5 * A' * D^-0.5, 其中 A' = A+I (如果self_loops=True)"""
    if self_loops:
        adj_prime = adj + torch.eye(adj.size(0), device=adj.device)
    else:
        adj_prime = adj
    
    degree = torch.sum(adj_prime, dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    # 处理度为0的节点 (如果存在)，防止 NaN/inf
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    degree_inv_sqrt[torch.isnan(degree_inv_sqrt)] = 0
    
    # 构建稀疏对角矩阵 D^-0.5
    # D_inv_sqrt_diag = torch.diag(degree_inv_sqrt)
    # return D_inv_sqrt_diag @ adj_prime @ D_inv_sqrt_diag

    # 如果 adj 是稀疏的，使用稀疏矩阵乘法会更有效
    # 为了简单和通用性，这里使用密集乘法，但注意如果 adj 非常大且稀疏，这不是最优的
    # PyTorch Geometric 的 GCNConv 会更高效地处理这些
    num_nodes = adj.size(0)
    if adj.is_sparse:
        # 构建稀疏的 D^-0.5
        indices = torch.arange(num_nodes, device=adj.device).unsqueeze(0).repeat(2,1)
        D_inv_sqrt_sparse = torch.sparse_coo_tensor(indices, degree_inv_sqrt, (num_nodes, num_nodes), device=adj.device)
        # D^-0.5 * A' * D^-0.5
        normalized_adj = torch.sparse.mm(D_inv_sqrt_sparse, adj_prime)
        normalized_adj = torch.sparse.mm(normalized_adj, D_inv_sqrt_sparse)
    else: # Dense case
        D_inv_sqrt_diag = torch.diag(degree_inv_sqrt)
        normalized_adj = D_inv_sqrt_diag @ adj_prime @ D_inv_sqrt_diag
    return normalized_adj

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn=None, dropout_rate=0.1, use_bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.activation_fn = activation_fn 
        self.dropout = nn.Dropout(dropout_rate)
        # self.reset_parameters() # 可选的权重初始化

    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.linear.weight)
    #     if self.linear.bias is not None:
    #         nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj_normalized):
        # x: (num_nodes, input_dim) - 节点特征
        # adj_normalized: (num_nodes, num_nodes) - 归一化邻接矩阵 (通常是 D^-0.5 * A' * D^-0.5)
        
        # 先对特征进行dropout和线性变换
        x_transformed = self.linear(self.dropout(x)) # WX^T or XW depending on convention, here XW
        
        # 执行图卷积: ÃXW
        # adj_normalized 是 D^-0.5 * A' * D^-0.5
        # x_transformed 是 XW
        if adj_normalized.is_sparse:
            output = torch.sparse.mm(adj_normalized, x_transformed)
        else:
            output = torch.mm(adj_normalized, x_transformed)
        
        if self.activation_fn:
            output = self.activation_fn(output)
        return output

class StockMixer(nn.Module):
    def __init__(self, stocks, time_steps, channels, no_graph_mixer_hidden_dim, scale, activation_fn_class=nn.Hardswish,
                 transformer_nhead=4, transformer_nlayers=2, transformer_dim_feedforward=128, transformer_dropout=0.1,
                 d_model_transformer=16,
                 gcn_dropout_rate=0.1, gcn_use_self_loops=True): # 新增 GCN 相关参数
        super(StockMixer, self).__init__()
        self.time_steps = time_steps
        self.channels = channels
        self.stocks = stocks # num_stocks
        self.d_model_transformer = d_model_transformer
        
        raw_scales_config = [
            {'type': 'conv', 'kernel_size': 2, 'stride': 2, 'time_steps': time_steps // 2},
            {'type': 'conv', 'kernel_size': 4, 'stride': 4, 'time_steps': time_steps // 4},
        ]
        current_scales_config = [s for s in raw_scales_config if s['time_steps'] > 0]
        self.multi_scale_time_mixer = MultiScaleTimeMixerRevised(
            initial_time_steps=time_steps,
            channels=self.channels,
            scales_config=current_scales_config,
            activation_fn_class=activation_fn_class
        )
        self.input_projection = nn.Linear(self.channels, self.d_model_transformer)
        _temp_T_concat = self.time_steps
        for scale_info in current_scales_config:
            if scale_info['type'] == 'conv':
                 _temp_T_concat += scale_info['time_steps']
        self.T_concat_for_pos_encoding = _temp_T_concat
        self.pos_encoder = PositionalEncoding(d_model=self.d_model_transformer, dropout=transformer_dropout, max_len=self.T_concat_for_pos_encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model_transformer,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)
        
        # GCN Layer (替换 NoGraphMixer)
        # GCN的输入维度是 d_model_transformer, 输出维度也是 d_model_transformer 以匹配 output_layer
        # 激活函数可以使用从 activation_fn_class 实例化的对象
        act_fn_instance = activation_fn_class() if isinstance(activation_fn_class, type) else activation_fn_class
        self.gcn_layer = GCNLayer(
            input_dim=self.d_model_transformer,
            output_dim=self.d_model_transformer, # 输出维度与输入一致，以便直接送入原 output_layer
            activation_fn=act_fn_instance,
            dropout_rate=gcn_dropout_rate
        )
        
        # 创建并注册归一化的邻接矩阵 (全连接图)
        # adj = torch.ones(self.stocks, self.stocks, device=torch.device('cpu')) # 先在CPU上创建
        # self.normalized_adj = normalize_adjacency_matrix(adj, self_loops=gcn_use_self_loops)
        # register_buffer 会将张量注册为模型的一部分，但不会被视为可训练参数
        # 它会被正确地移动到 GPU (如果模型 .to(device))
        # 注意: 邻接矩阵的创建应该在模型第一次 forward 调用前，或者在 __init__ 中如果 stocks 数量已知且固定。
        # 由于 stocks 是在初始化时传入的，我们可以在这里创建它。
        adj_raw = torch.ones(self.stocks, self.stocks) # 暂时不在特定设备创建，随模型移动
        # self.register_buffer('normalized_adj', normalize_adjacency_matrix(adj_raw, self_loops=gcn_use_self_loops))
        # 修正: register_buffer 需要张量，所以先归一化
        _normalized_adj_tensor = normalize_adjacency_matrix(adj_raw, self_loops=gcn_use_self_loops)
        self.register_buffer('normalized_adj', _normalized_adj_tensor)

        self.output_layer = nn.Linear(self.d_model_transformer, 1)

    def forward(self, x):
        # x 输入形状: (num_stocks, time_steps, channels)
        x_time_mixed = self.multi_scale_time_mixer(x)
        x_projected = self.input_projection(x_time_mixed)
        src_for_pos_encoding = x_projected.permute(1, 0, 2)
        src_pos_encoded = self.pos_encoder(src_for_pos_encoding)
        transformer_input = src_pos_encoded.permute(1, 0, 2)
        x_transformer_output_seq = self.transformer_encoder(transformer_input)
        x_aggregated_from_transformer = x_transformer_output_seq[:, -1, :]
        # x_aggregated_from_transformer shape: (num_stocks, self.d_model_transformer)

        # GCN 处理
        # normalized_adj 需要与 x_aggregated_from_transformer 在同一设备
        # register_buffer 应该处理了设备问题，但我们可以在forward中显式 .to(device) 以确保
        adj_for_gcn = self.normalized_adj # .to(x_aggregated_from_transformer.device) -> 已通过register_buffer处理
        x_gcn_output = self.gcn_layer(x_aggregated_from_transformer, adj_for_gcn)
        # x_gcn_output shape: (num_stocks, self.d_model_transformer)
        
        # 为了匹配 output_layer 和原始 NoGraphMixer 的输出习惯（带一个伪批次维度）
        # 我们给 GCN 的输出增加一个批次维度
        x_gcn_output_batched = x_gcn_output.unsqueeze(0) # (1, num_stocks, self.d_model_transformer)
        
        prediction = self.output_layer(x_gcn_output_batched)
        # prediction shape: (1, num_stocks, 1)
        return prediction.squeeze(-1).squeeze(0)


# class MemoryAugmentedStockMixer(nn.Module): # 暂时保持不变或做最小修改以避免错误
#     def __init__(self, stocks, time_steps, channels, no_graph_mixer_hidden_dim, 
#                  scale, 
#                  activation_fn_class=nn.Hardswish,
#                  memory_slots=64, memory_vector_dim_ratio=0.5, memory_read_head_num=4, memory_dropout=0.1,
#                  transformer_nhead=4, transformer_nlayers=2, transformer_dim_feedforward=128, transformer_dropout=0.1,
#                  d_model_transformer=16,
#                  gcn_dropout_rate=0.1, gcn_use_self_loops=True): # 添加 GCN 参数以匹配签名
#         super(MemoryAugmentedStockMixer, self).__init__()
#         self.stocks = stocks
#         self.time_steps = time_steps
#         self.channels = channels
#         self.memory_slots = memory_slots
#         self.d_model_transformer = d_model_transformer

#         # --- MultiScaleTimeMixerRevised 初始化 ---
#         raw_scales_config = [
#             {'type': 'conv', 'kernel_size': 2, 'stride': 2, 'time_steps': time_steps // 2},
#             {'type': 'conv', 'kernel_size': 4, 'stride': 4, 'time_steps': time_steps // 4},
#         ]
#         current_scales_config = [s for s in raw_scales_config if s['time_steps'] > 0]
        
#         self.multi_scale_time_mixer = MultiScaleTimeMixerRevised(
#             initial_time_steps=time_steps,
#             channels=self.channels,
#             scales_config=current_scales_config,
#             activation_fn_class=activation_fn_class
#         )
#         _temp_T_concat = self.time_steps
#         for scale_info in current_scales_config:
#             if scale_info['type'] == 'conv':
#                  _temp_T_concat += scale_info['time_steps']
#         self.T_concat_for_pos_encoding = _temp_T_concat

#         self.input_projection = nn.Linear(self.channels, self.d_model_transformer)
#         self.pos_encoder = PositionalEncoding(d_model=self.d_model_transformer, dropout=transformer_dropout, max_len=self.T_concat_for_pos_encoding)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.d_model_transformer,
#             nhead=transformer_nhead,
#             dim_feedforward=transformer_dim_feedforward,
#             dropout=transformer_dropout,
#             activation='relu',
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)

#         # --- 记忆模块初始化 (不变) ---
#         self.memory_vector_dim = int(self.d_model_transformer * memory_vector_dim_ratio)
#         if self.memory_vector_dim % memory_read_head_num != 0:
#             self.memory_vector_dim = (self.memory_vector_dim // memory_read_head_num) * memory_read_head_num
#             if self.memory_vector_dim == 0 and int(self.d_model_transformer * memory_vector_dim_ratio) > 0:
#                  self.memory_vector_dim = memory_read_head_num
#         if self.memory_vector_dim > 0 :
#             self.memory_bank = nn.Parameter(torch.randn(memory_slots, self.memory_vector_dim))
#             self.memory_read_attention = MemoryReadAttentionModule(
#                 query_dim=self.d_model_transformer,
#                 memory_vector_dim=self.memory_vector_dim,
#                 head_num=memory_read_head_num,
#                 dropout=memory_dropout
#             )
#             fusion_input_dim = self.d_model_transformer + self.memory_vector_dim
#         else: 
#             self.memory_bank = None
#             self.memory_read_attention = None
#             fusion_input_dim = self.d_model_transformer
#         if self.memory_vector_dim > 0:
#             self.fusion_mlp = nn.Sequential(
#                 nn.Linear(fusion_input_dim, self.d_model_transformer), 
#                 activation_fn_class(),
#                 nn.Dropout(memory_dropout)
#             )
#             mixer_input_dim = self.d_model_transformer # GCN/NoGraphMixer 的输入维度
#         else:
#             self.fusion_mlp = None
#             mixer_input_dim = self.d_model_transformer
        
#         # --- GCN Layer (或 NoGraphMixer 的占位) ---
#         # 为了避免 MemoryAugmentedStockMixer 出错，暂时还用 NoGraphMixer
#         # 或者可以也改成GCN，但需要确保逻辑一致
#         self.stock_interaction_mixer = NoGraphMixer( # 保持原样
#             input_dim=mixer_input_dim, 
#             hidden_dim=no_graph_mixer_hidden_dim,
#             activation_fn_class=activation_fn_class
#         )
#         self.output_layer = nn.Linear(mixer_input_dim, 1)

#     def forward(self, x):
#         # x 输入形状: (num_stocks, time_steps, channels)
#         x_time_mixed = self.multi_scale_time_mixer(x)
#         x_projected = self.input_projection(x_time_mixed)
#         src_for_pos_encoding = x_projected.permute(1, 0, 2)
#         src_pos_encoded = self.pos_encoder(src_for_pos_encoding)
#         transformer_input = src_pos_encoded.permute(1, 0, 2)
#         x_transformer_output_seq = self.transformer_encoder(transformer_input)
#         x_transformer_aggregated = x_transformer_output_seq[:, -1, :]
        
#         current_batch_size_for_memory = x_transformer_aggregated.size(0)
#         if self.memory_vector_dim > 0 and self.memory_read_attention is not None and self.fusion_mlp is not None:
#             memory_output, _ = self.memory_read_attention(x_transformer_aggregated, self.memory_bank.unsqueeze(0).repeat(current_batch_size_for_memory, 1, 1))
#             fused_input = torch.cat([x_transformer_aggregated, memory_output], dim=1)
#             processed_features = self.fusion_mlp(fused_input)
#         else:
#             processed_features = x_transformer_aggregated

#         # 使用 stock_interaction_mixer (即 NoGraphMixer)
#         x_stock_features_for_mixer = processed_features.unsqueeze(0)
#         x_graph_mixed = self.stock_interaction_mixer(x_stock_features_for_mixer)
        
#         prediction = self.output_layer(x_graph_mixed)
#         return prediction.squeeze(-1).squeeze(0)

# 为了让文件能被解析，我将暂时完整注释掉 MemoryAugmentedStockMixer 类
# 因为 train.py 目前只使用 StockMixer，这样可以避免因参数不匹配导致的导入错误
# 后续如果需要，可以再解除注释并适配 GCN
class MemoryAugmentedStockMixer(nn.Module):
    pass # 暂时将其变为空壳

