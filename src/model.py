import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from typing import Dict, List, Tuple, Optional

acv = nn.GELU()

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    
    # 保守的形状检查和修复：只在检测到可能的广播问题时进行修复
    original_pred_shape = prediction.shape
    original_base_shape = base_price.shape
    
    # 检查是否可能出现广播问题
    if prediction.ndim == 1 and len(base_price.shape) == 2 and base_price.shape[1] == 1:
        # 这种情况下会发生 (N,) vs (N,1) 的广播，转换为 (N,1) vs (N,1)
        prediction = prediction.unsqueeze(1)
        print(f"信息：检测到潜在的广播问题，已修复预测张量形状: {original_pred_shape} -> {prediction.shape}")
    elif prediction.ndim == 2 and prediction.shape != base_price.shape:
        # 其他可能的形状不匹配情况
        if prediction.shape[0] == base_price.shape[0] and prediction.shape[1] != 1:
            print(f"警告：预测张量形状异常: {original_pred_shape}，base_price形状: {original_base_shape}")
            prediction = prediction.squeeze().unsqueeze(1)
            print(f"已尝试修复为: {prediction.shape}")
    
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
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
    def __init__(self, initial_time_steps, channels, scales_config, 
                 activation_fn_class=nn.Hardswish, use_triu_network=True):
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
                              kernel_size=scale_info['kernel'], stride=scale_info['stride'])
                )
                self.conv_scale_lns.append(nn.LayerNorm([ts_after_conv, channels]))
                concat_time_dim_for_triu += ts_after_conv
            elif scale_info['type'] == 'original':
                # This part of config is mainly for calculating concat_time_dim_for_triu
                # and for clarity. Original scale features are handled directly from 'inputs'.
                pass 

        if concat_time_dim_for_triu == 0:
            raise ValueError("MultiScaleTimeMixer: concat_time_dim_for_triu is 0. No valid scales.")

        if use_triu_network:
            # print("  - 构建时序网络: 使用 TriU 网络") # 在train脚本中打印，这里保持干净
            triu_network = nn.Sequential(
                TriU(concat_time_dim_for_triu),
                activation_fn_class(),
                TriU(concat_time_dim_for_triu)
            )
        else:
            # print("  - [消融实验] 构建时序网络: 使用标准线性层 (Linear) 替代 TriU")
            triu_network = nn.Sequential(
                nn.Linear(concat_time_dim_for_triu, concat_time_dim_for_triu),
                activation_fn_class(),
                nn.Linear(concat_time_dim_for_triu, concat_time_dim_for_triu)
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


class StockPredict(nn.Module):
    def __init__(self, stocks, time_steps, channels, scale, activation_fn_class=nn.Hardswish, 
                 attention_num_heads=4, attention_hidden_ff_dim=None, attention_dropout_rate=0.1,
                 use_sparse_attention=True, use_multiscale_fusion=True, use_triu_network=True):
        super(StockPredict, self).__init__()
        
        # 存储股票数量和消融实验标志
        self.stocks = stocks
        self.use_sparse_attention = use_sparse_attention
        
        # --- [消融实验] 多尺度融合模块 ---
        if not use_multiscale_fusion:
            # print("  - [消融实验] 已禁用多尺度时间融合，scale_factor 将被强制设为 1")
            scale = 1
        # --------------------------------

        current_scales_config_for_mixer = [] 
        calculated_concat_time_dim = 0 

        # 1. Original Scale
        current_scales_config_for_mixer.append({
            'type': 'original', 
            'time_steps': time_steps 
        })
        calculated_concat_time_dim += time_steps

        # 2. Convolutional Scales
        num_conv_scales_to_add = max(0, scale - 1)
        for i in range(num_conv_scales_to_add):
            current_stride = 2**(i + 1)
            current_kernel = 2**(i + 1)
            ts_after_conv = time_steps // current_stride
            if ts_after_conv >= 1:
                current_scales_config_for_mixer.append({
                    'type': 'conv',
                    'kernel': current_kernel,
                    'stride': current_stride,
                    'time_steps': ts_after_conv 
                })
                calculated_concat_time_dim += ts_after_conv
            else:
                break
        
        if not current_scales_config_for_mixer or calculated_concat_time_dim == 0:
            raise ValueError(
                f"StockPredict init: No valid scales generated. "
                f"Initial time_steps: {time_steps}, requested total scales: {scale}. "
                f"Resulting concat_time_dim: {calculated_concat_time_dim}. "
                "Ensure time_steps is large enough for the requested number of scales."
            )
        
        self.temporal_mixer = MultiScaleTimeMixer(
            initial_time_steps=time_steps,
            channels=channels,
            scales_config=current_scales_config_for_mixer,
            activation_fn_class=activation_fn_class,
            use_triu_network=use_triu_network # 传递标志
        )
        
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(calculated_concat_time_dim, 1)
        
        # Determine hidden_ff_dim for StockAttentionMixer if not provided
        # A common practice is to make it 4 * embed_dim, but can be tuned
        if attention_hidden_ff_dim is None:
            attention_hidden_ff_dim = calculated_concat_time_dim * 4 

        self.stock_attention_mixer = StockAttentionMixer(
            embed_dim=calculated_concat_time_dim,
            num_heads=attention_num_heads,
            hidden_ff_dim=attention_hidden_ff_dim,
            dropout_rate=attention_dropout_rate,
            activation_fn_class=activation_fn_class
        )
        
        self.time_fc_ = nn.Linear(calculated_concat_time_dim, 1)
        
        # 稀疏注意力配置状态
        self.sparse_attention_configured = False

    def setup_sparse_attention(self, dataset_name: str, ticker_list: list):
        """
        设置模型的行业稀疏注意力
        
        Args:
            dataset_name: 数据集名称 (SP500, NASDAQ, NYSE)
            ticker_list: 股票代码列表（按数据集顺序）
        """
        # --- [消融实验] 稀疏注意力模块 ---
        if not self.use_sparse_attention:
            # print("  - [消融实验] 已禁用稀疏注意力，将使用标准全连接注意力")
            self.sparse_attention_configured = False # 确保forward pass中使用全连接
            return
        # --------------------------------

        print(f"🎯 为StockPredict模型设置行业稀疏注意力...")
        print(f"   数据集: {dataset_name}")
        print(f"   ticker_list长度: {len(ticker_list)}")
        print(f"   模型股票数量: {self.stocks}")
        
        # 设置StockAttentionMixer的稀疏注意力，传递实际股票数量
        self.stock_attention_mixer.setup_sparse_attention(dataset_name, ticker_list, self.stocks)
        self.sparse_attention_configured = True
        
        print(f"✅ StockPredict模型稀疏注意力设置完成")

    def forward(self, inputs):
        # inputs: (batch_size_is_stock_num, time_steps, channels)
        y_temporal_mixed = self.temporal_mixer(inputs)
        # y_temporal_mixed: (stock_num, calculated_concat_time_dim, channels)
        
        y_channel_processed = self.channel_fc(y_temporal_mixed).squeeze(-1)
        # y_channel_processed: (stock_num, calculated_concat_time_dim)
        
        y_out = self.time_fc(y_channel_processed)
        # y_out: (stock_num, 1)
        
        z_stock_mixed = self.stock_attention_mixer(y_channel_processed)
        # z_stock_mixed: (stock_num, calculated_concat_time_dim)
        
        z_out = self.time_fc_(z_stock_mixed)
        # z_out: (stock_num, 1)
        
        return y_out + z_out


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


class IndustrySparseAttention(nn.Module):
    """
    基于行业分类的稀疏注意力机制
    
    核心思想：
    - 只计算同行业股票之间的注意力权重
    - 跨行业股票的注意力权重设为0
    - 大幅降低计算复杂度：从O(n²)降为O(k²×m)，其中k=平均每行业股票数，m=行业数
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"
        
        # 标准的注意力投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # 存储行业掩码矩阵
        self.register_buffer('industry_mask', None)
        self.industry_mapping = None
        
    def create_industry_mask(self, industry_mapping: Dict[str, str], ticker_list: List[str]) -> torch.Tensor:
        """
        根据行业映射创建稀疏注意力掩码矩阵
        
        Args:
            industry_mapping: {ticker: sic_description} 映射
            ticker_list: 股票代码列表
            
        Returns:
            掩码矩阵 [num_stocks, num_stocks]，同行业为1，跨行业为0
        """
        num_stocks = len(ticker_list)
        mask = torch.zeros(num_stocks, num_stocks, dtype=torch.bool)
        
        # 创建ticker到索引的映射
        ticker_to_idx = {ticker: idx for idx, ticker in enumerate(ticker_list)}
        
        # 按行业分组
        industry_groups = {}
        missing_industry_count = 0
        
        for ticker in ticker_list:
            if ticker in industry_mapping:
                industry = industry_mapping[ticker]
                if industry not in industry_groups:
                    industry_groups[industry] = []
                industry_groups[industry].append(ticker_to_idx[ticker])
            else:
                # 对于缺失行业信息的股票，创建独立的"未知行业"组
                unknown_industry = f"UNKNOWN_INDUSTRY_{missing_industry_count}"
                industry_groups[unknown_industry] = [ticker_to_idx[ticker]]
                missing_industry_count += 1
                
        print(f"📊 掩码创建统计:")
        print(f"   总股票数: {num_stocks}")
        print(f"   有行业信息: {num_stocks - missing_industry_count}")
        print(f"   缺失行业信息: {missing_industry_count}")
        print(f"   行业组数: {len(industry_groups)}")
        
        # 为每个行业内的股票设置掩码
        for industry, indices in industry_groups.items():
            for i in indices:
                for j in indices:
                    mask[i, j] = True
        
        return mask
    
    def load_industry_data(self, dataset_name: str) -> Dict[str, str]:
        """
        加载指定数据集的行业数据
        
        Args:
            dataset_name: 数据集名称 (SP500, NASDAQ, NYSE)
            
        Returns:
            {ticker: sic_description} 映射字典
        """
        json_path = f'../dataset/{dataset_name}/{dataset_name.lower()}_industry_data.json'
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            industry_mapping = {}
            for ticker, info in data.items():
                if 'error' not in info and 'sic_description' in info:
                    industry_mapping[ticker] = info['sic_description']
                    
            print(f"✅ 成功加载 {dataset_name} 行业数据: {len(industry_mapping)} 只股票")
            return industry_mapping
            
        except FileNotFoundError:
            print(f"❌ 未找到文件: {json_path}")
            return {}
        except Exception as e:
            print(f"❌ 加载行业数据失败: {e}")
            return {}
    
    def setup_sparse_attention(self, dataset_name: str, ticker_list: List[str], actual_stock_num: int):
        """
        设置稀疏注意力掩码
        
        Args:
            dataset_name: 数据集名称
            ticker_list: 股票代码列表（按数据集顺序）
            actual_stock_num: 实际的股票数量（从模型获取）
        """
        print(f"🎯 设置稀疏注意力掩码...")
        print(f"   ticker_list长度: {len(ticker_list)}")
        print(f"   模型股票数量: {actual_stock_num}")
        
        # 调整ticker_list以匹配实际股票数量
        if len(ticker_list) != actual_stock_num:
            if len(ticker_list) < actual_stock_num:
                # 如果ticker_list较短，用占位符填充
                diff = actual_stock_num - len(ticker_list)
                ticker_list = ticker_list + [f"PLACEHOLDER_{i}" for i in range(diff)]
                print(f"✅ 已用{diff}个占位符扩展ticker_list到{len(ticker_list)}")
            else:
                # 如果ticker_list较长，截断
                ticker_list = ticker_list[:actual_stock_num]
                print(f"✅ 已截断ticker_list到{len(ticker_list)}")
        
        # 加载行业映射
        self.industry_mapping = self.load_industry_data(dataset_name)
        
        if not self.industry_mapping:
            print("⚠️ 警告: 未能加载行业数据，将使用全连接注意力")
            # 创建全连接掩码
            self.industry_mask = torch.ones(actual_stock_num, actual_stock_num, dtype=torch.bool)
        else:
            # 创建稀疏掩码
            self.industry_mask = self.create_industry_mask(self.industry_mapping, ticker_list)
            
            # 统计稀疏度
            total_connections = self.industry_mask.numel()
            active_connections = self.industry_mask.sum().item()
            sparsity = 1 - (active_connections / total_connections)
            
            print(f"🎯 稀疏注意力设置完成:")
            print(f"   掩码大小: {self.industry_mask.shape}")
            print(f"   总连接数: {total_connections:,}")
            print(f"   激活连接数: {active_connections:,}")
            print(f"   稀疏度: {sparsity:.2%}")
            
            # 分析行业分布
            self._analyze_industry_distribution(ticker_list)
    
    def _analyze_industry_distribution(self, ticker_list: List[str]):
        """分析行业分布统计"""
        if not self.industry_mapping:
            return
            
        industry_counts = {}
        for ticker in ticker_list:
            if ticker in self.industry_mapping:
                industry = self.industry_mapping[ticker]
                industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        print(f"📊 行业分布 (前10个):")
        sorted_industries = sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)
        for industry, count in sorted_industries[:10]:
            industry_short = industry[:50] + "..." if len(industry) > 50 else industry
            print(f"   {industry_short}: {count} 只股票")
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            attention_mask: 额外的注意力掩码
            
        Returns:
            输出张量 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 投影到Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在形状: [batch_size, num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 形状: [batch_size, num_heads, seq_len, seq_len]
        
        # 应用行业稀疏掩码
        if self.industry_mask is not None:
            # 确保掩码在正确设备上
            industry_mask = self.industry_mask.to(attn_scores.device)
            
            # 扩展掩码维度以匹配多头注意力
            industry_mask_expanded = industry_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            industry_mask_expanded = industry_mask_expanded.expand(batch_size, self.num_heads, -1, -1)
            
            # 将跨行业的注意力分数设为负无穷
            attn_scores = attn_scores.masked_fill(~industry_mask_expanded, float('-inf'))
        
        # 应用额外掩码（如果提供）
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑回原始形状
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 输出投影
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out


class StockAttentionMixer(nn.Module):
    """
    基于行业稀疏注意力的股票混合器
    使用Polygon API收集的行业数据实现稀疏注意力，只允许同行业股票之间的注意力交互
    """
    def __init__(self, embed_dim, num_heads, hidden_ff_dim, dropout_rate=0.1, activation_fn_class=nn.GELU):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 使用行业稀疏注意力替代标准多头注意力
        self.sparse_attention = IndustrySparseAttention(embed_dim, num_heads, dropout_rate)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_ff_dim),
            activation_fn_class(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_ff_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 存储是否已设置稀疏注意力
        self.sparse_attention_configured = False

    def setup_sparse_attention(self, dataset_name: str, ticker_list: list, actual_stock_num: int):
        """
        设置行业稀疏注意力掩码
        
        Args:
            dataset_name: 数据集名称 (SP500, NASDAQ, NYSE)
            ticker_list: 股票代码列表（按数据集顺序）
            actual_stock_num: 实际的股票数量
        """
        print(f"🎯 为StockAttentionMixer设置行业稀疏注意力...")
        self.sparse_attention.setup_sparse_attention(dataset_name, ticker_list, actual_stock_num)
        self.sparse_attention_configured = True
        print(f"✅ StockAttentionMixer稀疏注意力设置完成")

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [stock_num, embed_dim]
            
        Returns:
            输出张量 [stock_num, embed_dim]
        """
        # 调整输入形状以适应稀疏注意力
        # 从 (stock_num, embed_dim) 转换为 (1, stock_num, embed_dim)
        batch_size = 1
        seq_len = x.shape[0]
        x_batched = x.unsqueeze(0)  # (1, stock_num, embed_dim)
        
        # 残差连接 + 稀疏注意力
        residual = x_batched
        x_norm = self.norm1(x_batched)
        attn_output = self.sparse_attention(x_norm)  # (1, stock_num, embed_dim)
        x_res1 = residual + attn_output
        
        # 残差连接 + 前馈网络
        residual = x_res1
        x_norm2 = self.norm2(x_res1)
        ffn_output = self.ffn(x_norm2)  # (1, stock_num, embed_dim)
        x_res2 = residual + ffn_output
        
        # 转换回原始形状 (stock_num, embed_dim)
        return x_res2.squeeze(0)

