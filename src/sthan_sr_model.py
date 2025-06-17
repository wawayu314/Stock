import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalAttention(nn.Module):
    """时间注意力机制"""
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_query_vector = nn.Parameter(torch.Tensor(hidden_dim)) # v_a
        self.linear_transform = nn.Linear(hidden_dim, hidden_dim, bias=True) # W_a * h_i + b_a
        
        nn.init.xavier_uniform_(self.attention_query_vector.data.unsqueeze(1))
        # nn.Linear has default init

    def forward(self, gru_outputs):
        # gru_outputs: (batch_size_is_num_stocks, seq_len, gru_hidden_dim)
        
        # u_i = tanh(W_a * h_i + b_a)
        u_i = torch.tanh(self.linear_transform(gru_outputs)) # (batch, seq_len, hidden_dim)
        
        # e_i = v_a^T * u_i
        e_i = torch.matmul(u_i, self.attention_query_vector) # (batch, seq_len)
        
        # alpha_i = softmax(e_i)
        alpha_i = F.softmax(e_i, dim=1) # (batch, seq_len)
        
        # c = sum(alpha_i * h_i)
        context_vector = torch.sum(alpha_i.unsqueeze(2) * gru_outputs, dim=1) # (batch, hidden_dim)
        
        return context_vector, alpha_i

class SpatialAttention(nn.Module):
    """空间注意力机制 (股票间多头自注意力)"""
    def __init__(self, input_dim, num_heads=4, dropout_rate=0.1):
        super(SpatialAttention, self).__init__()
        if input_dim % num_heads != 0:
            raise ValueError("input_dim 必须能被 num_heads 整除")
            
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.query_layer = nn.Linear(input_dim, input_dim) # W_q * X
        self.key_layer = nn.Linear(input_dim, input_dim)   # W_k * X
        self.value_layer = nn.Linear(input_dim, input_dim) # W_v * X
        
        self.fc_out = nn.Linear(input_dim, input_dim) 
        self.dropout = nn.Dropout(dropout_rate)
        # self.scale_factor will be computed in forward pass based on K.device

    def forward(self, x_temporal_contexts):
        # x_temporal_contexts: (N, input_dim) where N = num_stocks
        N = x_temporal_contexts.shape[0]
        
        Q = self.query_layer(x_temporal_contexts) # (N, input_dim)
        K = self.key_layer(x_temporal_contexts)   # (N, input_dim)
        V = self.value_layer(x_temporal_contexts)   # (N, input_dim)

        # Reshape for multi-head: (N, num_heads, head_dim)
        Q = Q.view(N, self.num_heads, self.head_dim)
        K = K.view(N, self.num_heads, self.head_dim)
        V = V.view(N, self.num_heads, self.head_dim)

        # Transpose for matmul: Q (num_heads, N, head_dim), K (num_heads, head_dim, N)
        Q = Q.permute(1, 0, 2) 
        K_T = K.permute(1, 2, 0) # K.T for matmul
        V = V.permute(1, 0, 2) 
        
        scale_factor = torch.sqrt(torch.FloatTensor([self.head_dim])).to(K.device)

        # energy = (num_heads, N, head_dim) @ (num_heads, head_dim, N) -> (num_heads, N, N)
        energy = torch.matmul(Q, K_T) / scale_factor
            
        attention_weights = torch.softmax(energy, dim=-1) # (num_heads, N, N)
        attention_weights = self.dropout(attention_weights)
        
        # weighted_values = (num_heads, N, N) @ (num_heads, N, head_dim) -> (num_heads, N, head_dim)
        weighted_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        # (num_heads, N, head_dim) -> (N, num_heads, head_dim) -> (N, num_heads * head_dim = input_dim)
        weighted_values = weighted_values.permute(1, 0, 2).contiguous().view(N, self.input_dim)
        
        output = self.fc_out(weighted_values) # (N, input_dim)
        
        return output, attention_weights

class STHAN_SR_Predictor(nn.Module):
    def __init__(self, input_feature_dim, num_stocks, # num_stocks is for info, not direct layer sizing
                 gru_hidden_dim=64, gru_num_layers=1, gru_dropout_rate=0.1,
                 spatial_attention_heads=4, spatial_attention_dropout_rate=0.1,
                 output_dim=1, use_tanh_output=True):
        super(STHAN_SR_Predictor, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.use_tanh_output = use_tanh_output

        # 1. GRU Layer for temporal encoding
        self.gru = nn.GRU(
            input_size=input_feature_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True, # expects (batch, seq, feature)
            dropout=gru_dropout_rate if gru_num_layers > 1 else 0.0
        )
        
        # 2. Temporal Attention Layer
        self.temporal_attention = TemporalAttention(hidden_dim=gru_hidden_dim)
        
        # 3. Spatial Attention Layer
        # Input to spatial attention is the output of temporal attention (gru_hidden_dim)
        self.spatial_attention = SpatialAttention(
            input_dim=gru_hidden_dim, 
            num_heads=spatial_attention_heads,
            dropout_rate=spatial_attention_dropout_rate
        )
        
        # 4. Prediction Layer
        # Input to prediction layer is the output of spatial attention (gru_hidden_dim)
        self.prediction_layer = nn.Linear(gru_hidden_dim, output_dim)

    def forward(self, x):
        # x: (num_stocks, lookback_length, fea_num)
        # In our case, num_stocks is effectively the batch_size for GRU and TemporalAttention
        
        # 1. GRU encoding
        # gru_out: (num_stocks, lookback_length, gru_hidden_dim)
        # h_n: (num_layers, num_stocks, gru_hidden_dim) - final hidden state
        gru_out, _ = self.gru(x) 
        
        # 2. Temporal Attention
        # temporal_context: (num_stocks, gru_hidden_dim)
        temporal_context, _ = self.temporal_attention(gru_out)
        
        # 3. Spatial Attention
        # spatial_output: (num_stocks, gru_hidden_dim)
        # The input to spatial attention is the context from temporal attention.
        # It treats each stock's temporal context as an item in a sequence for spatial self-attention.
        spatial_output, _ = self.spatial_attention(temporal_context)
        
        # 4. Prediction
        prediction = self.prediction_layer(spatial_output) # (num_stocks, output_dim)
        
        if self.use_tanh_output:
            return torch.tanh(prediction)
        return prediction

if __name__ == '__main__':
    # 测试代码
    num_stocks_test = 10
    lookback_len_test = 16
    feature_dim_test = 5
    device_test = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Testing on device: {device_test}")

    model_test = STHAN_SR_Predictor(
        input_feature_dim=feature_dim_test,
        num_stocks=num_stocks_test,
        gru_hidden_dim=32,
        gru_num_layers=1,
        gru_dropout_rate=0.0, # No dropout for single layer GRU
        spatial_attention_heads=2,
        spatial_attention_dropout_rate=0.1,
        output_dim=1,
        use_tanh_output=True
    ).to(device_test)
    
    model_test.train() # Set to train mode for dropout if any

    dummy_input = torch.randn(num_stocks_test, lookback_len_test, feature_dim_test).to(device_test)
    print(f"Input shape: {dummy_input.shape}")
    
    output = model_test(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: (num_stocks, 1)
    assert output.shape == (num_stocks_test, 1)
    print(f"Output device: {output.device}")

    print("\nModel Structure:")
    print(model_test)

    total_params = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # Check if backward pass works
    target = torch.randn(num_stocks_test, 1).to(device_test)
    loss = F.mse_loss(output, target)
    loss.backward()
    print("\nBackward pass successful.")

    # Test individual components if needed (example for TemporalAttention)
    temp_att_test = TemporalAttention(hidden_dim=32).to(device_test)
    gru_out_dummy = torch.randn(num_stocks_test, lookback_len_test, 32).to(device_test)
    temp_context_test, _ = temp_att_test(gru_out_dummy)
    print(f"\nTemporal Attention test - Context shape: {temp_context_test.shape}, Device: {temp_context_test.device}")
    assert temp_context_test.shape == (num_stocks_test, 32)

    # Test individual components if needed (example for SpatialAttention)
    spatial_att_test = SpatialAttention(input_dim=32, num_heads=2).to(device_test)
    spatial_input_dummy = torch.randn(num_stocks_test, 32).to(device_test)
    spatial_output_test, _ = spatial_att_test(spatial_input_dummy)
    print(f"Spatial Attention test - Output shape: {spatial_output_test.shape}, Device: {spatial_output_test.device}")
    assert spatial_output_test.shape == (num_stocks_test, 32)
    
    print("\nSTHAN-SR Model Test Completed.") 