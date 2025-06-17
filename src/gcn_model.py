import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_features, adj_matrix_normalized):
        # input_features: (N, in_features) N is number of stocks
        # adj_matrix_normalized: (N, N) D^-0.5 * A_hat * D^-0.5
        
        support = torch.mm(input_features, self.weight) # H * W
        output = torch.spmm(adj_matrix_normalized, support) # adj * H * W (sparse matrix multiplication)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNStockPredictor(nn.Module):
    def __init__(self, num_stocks, input_feature_dim, num_gcn_layers, gcn_hidden_dims, 
                 output_dim=1, dropout_rate=0.1, activation_fn_class=nn.ReLU, use_tanh_output=True):
        """
        GCN model for stock prediction.
        Operates on features from the last time step of the input sequence.

        Args:
            num_stocks (int): Total number of stocks (N).
            input_feature_dim (int): Dimension of input features per stock (fea_num).
            num_gcn_layers (int): Number of GCN layers.
            gcn_hidden_dims (list of int): List of output dimensions for each GCN layer.
                                         Length should be num_gcn_layers.
            output_dim (int): Final output dimension (usually 1).
            dropout_rate (float): Dropout rate.
            activation_fn_class (nn.Module): Activation function class to use between GCN layers (e.g., nn.ReLU).
            use_tanh_output (bool): Whether to apply tanh to the final output.
        """
        super(GCNStockPredictor, self).__init__()

        if not (num_gcn_layers == len(gcn_hidden_dims)):
            raise ValueError("num_gcn_layers and len(gcn_hidden_dims) must be equal.")

        self.num_stocks = num_stocks
        self.dropout_rate = dropout_rate
        self.use_tanh_output = use_tanh_output

        self.gcn_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        current_dim = input_feature_dim
        for i in range(num_gcn_layers):
            self.gcn_layers.append(GCNLayer(current_dim, gcn_hidden_dims[i]))
            self.activations.append(activation_fn_class())
            if i < num_gcn_layers -1 : # No dropout after the last GCN layer's activation before FC
                 self.dropout_layers.append(nn.Dropout(dropout_rate))
            current_dim = gcn_hidden_dims[i]

        self.final_fc = nn.Linear(current_dim, output_dim)
        
        # Precompute normalized adjacency matrix for a fully connected graph with self-loops
        adj = torch.ones(num_stocks, num_stocks) # A_hat = A + I (implicitly, as A is all ones for fully connected)
        # For a fully connected graph (all 1s), D_ii = N. D_inv_sqrt_ii = 1/sqrt(N)
        # D_inv_sqrt = torch.eye(num_stocks) * (1.0 / torch.sqrt(torch.tensor(float(num_stocks))))
        # adj_normalized_dense = D_inv_sqrt @ adj @ D_inv_sqrt 
        # However, for GCN, the normalization is usually done based on degrees of A_hat = A + I
        # Let's use the standard GCN normalization:
        # A_hat = A + I. For fully connected A, A_hat is all ones.
        # D_hat_ii = sum_j A_hat_ij. For all ones A_hat, D_hat_ii = num_stocks.
        
        # Let A be the adjacency matrix without self-loops for a fully connected graph
        adj_no_self_loops = torch.ones(num_stocks, num_stocks) - torch.eye(num_stocks)
        A_hat = adj_no_self_loops + torch.eye(num_stocks) # Add self-loops
        
        D_hat_diag = torch.sum(A_hat, dim=1)
        D_hat_diag_inv_sqrt = torch.pow(D_hat_diag, -0.5)
        D_hat_diag_inv_sqrt[torch.isinf(D_hat_diag_inv_sqrt)] = 0. # Handle isolated nodes if any (not for fully connected)
        
        # Efficiently create sparse D_inv_sqrt matrix
        # Or, for dense multiplication which spmm supports:
        D_inv_sqrt_matrix = torch.diag(D_hat_diag_inv_sqrt)
        
        # adj_normalized = D_inv_sqrt @ A_hat @ D_inv_sqrt
        # For dense adj_matrix_normalized:
        adj_normalized_dense = D_inv_sqrt_matrix @ A_hat @ D_inv_sqrt_matrix
        
        # For torch.spmm, the adjacency matrix needs to be sparse.
        # Let's convert adj_normalized_dense to a sparse COO tensor.
        # It's often better if the original A_hat is sparse, but for a dense fully connected graph,
        # making it sparse and then doing spmm might not be more efficient than dense mm if N is not huge.
        # Given stock_num ~1000, dense mm might be acceptable.
        # Let's stick to torch.mm in GCNLayer if adj_matrix_normalized is dense,
        # or ensure adj_matrix_normalized is sparse for torch.spmm.
        # For simplicity with a fully connected graph, let's assume GCNLayer can handle dense adj.
        # And we will use torch.mm in GCNLayer and pass dense normalized adj.
        # We need to modify GCNLayer to use torch.mm for adj part if adj is dense.
        # Or, make adj_normalized_dense sparse here.
        
        # Let's prepare a dense normalized adjacency matrix and register it as a buffer.
        # This adj_matrix is D^-0.5 * (A+I) * D^-0.5
        self.register_buffer('adj_matrix_normalized', adj_normalized_dense)


    def forward(self, x):
        # x: input (batch_size_is_stock_num, lookback_length, fea_num)
        # For GCN, N is stock_num, which is x.size(0)
        # We need to ensure num_stocks passed during init matches x.size(0) in forward pass
        if x.size(0) != self.num_stocks:
            raise ValueError(f"Input batch size {x.size(0)} (num_stocks) does not match model's num_stocks {self.num_stocks}")

        # Use features from the last time step as initial node features for GCN
        # node_features: (stock_num, fea_num)
        h = x[:, -1, :] 

        for i, gcn_layer in enumerate(self.gcn_layers):
            # The GCNLayer expects dense adj_matrix for torch.mm
            # If GCNLayer uses torch.spmm, then adj_matrix_normalized should be sparse
            # Current GCNLayer uses torch.spmm, so adj_matrix_normalized needs to be sparse
            # Let's adjust GCNLayer or the adj_matrix preparation.
            # Easiest for now: modify GCNLayer to use torch.mm if adj is dense.
            # Or, make self.adj_matrix_normalized sparse.
            # Let's assume self.adj_matrix_normalized is kept dense, and GCNLayer will use torch.mm.
            # I will modify GCNLayer below to reflect this.
            
            h = gcn_layer(h, self.adj_matrix_normalized)
            h = self.activations[i](h)
            if i < len(self.dropout_layers): # Apply dropout if available for this layer
                h = self.dropout_layers[i](h)
        
        raw_prediction = self.final_fc(h) # (stock_num, output_dim)
        
        if self.use_tanh_output:
            prediction = torch.tanh(raw_prediction)
        else:
            prediction = raw_prediction
            
        return prediction

# Modify GCNLayer to use torch.mm for the adjacency matrix multiplication,
# assuming adj_matrix_normalized is dense.
GCNLayer._original_forward = GCNLayer.forward
def gcn_layer_forward_dense_adj(self, input_features, adj_matrix_normalized_dense):
    support = torch.mm(input_features, self.weight) # H * W
    # Use torch.mm for dense adjacency matrix
    output = torch.mm(adj_matrix_normalized_dense, support) # adj * H * W 
    if self.bias is not None:
        return output + self.bias
    else:
        return output
GCNLayer.forward = gcn_layer_forward_dense_adj 