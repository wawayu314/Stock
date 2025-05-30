import torch
import torch.nn as nn
import torch.nn.functional as F

acv = nn.GELU()

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
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
                              kernel_size=scale_info['kernel'], stride=scale_info['stride'])
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


class StockMixer(nn.Module):
    def __init__(self, stocks, time_steps, channels, market, scale, activation_fn_class=nn.Hardswish):
        super(StockMixer, self).__init__()

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
                f"StockMixer init: No valid scales generated. "
                f"Initial time_steps: {time_steps}, requested total scales: {scale}. "
                f"Resulting concat_time_dim: {calculated_concat_time_dim}. "
                "Ensure time_steps is large enough for the requested number of scales."
            )
        
        self.temporal_mixer = MultiScaleTimeMixerRevised(
            initial_time_steps=time_steps,
            channels=channels,
            scales_config=current_scales_config_for_mixer,
            activation_fn_class=activation_fn_class
        )
        
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(calculated_concat_time_dim, 1)
        self.stock_mixer = NoGraphMixer(stocks, market) 
        self.time_fc_ = nn.Linear(calculated_concat_time_dim, 1)

    def forward(self, inputs):
        # inputs: (batch_size_is_stock_num, time_steps, channels)
        y_temporal_mixed = self.temporal_mixer(inputs)
        # y_temporal_mixed: (stock_num, calculated_concat_time_dim, channels)
        
        y_channel_processed = self.channel_fc(y_temporal_mixed).squeeze(-1)
        # y_channel_processed: (stock_num, calculated_concat_time_dim)
        
        y_out = self.time_fc(y_channel_processed)
        # y_out: (stock_num, 1)
        
        z_stock_mixed = self.stock_mixer(y_channel_processed) 
        # z_stock_mixed: (stock_num, calculated_concat_time_dim)
        
        z_out = self.time_fc_(z_stock_mixed)
        # z_out: (stock_num, 1)
        
        return y_out + z_out

