import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, negative_slope=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Linear transformation weight
        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        # Attention mechanism: one vector per head
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # x shape: [N, in_channels]
        N = x.size(0)
        # Linearly transform and reshape for multi-head attention
        x = torch.matmul(x, self.weight)  # [N, heads*out_channels]
        x = x.view(N, self.heads, self.out_channels)  # [N, heads, out_channels]

        # edge_index: [2, E] with edge_index[0]=source, edge_index[1]=target
        src, dst = edge_index

        # Prepare attention computation: for each edge we need [x_src || x_dst]
        x_src = x[src]  # [E, heads, out_channels]
        x_dst = x[dst]  # [E, heads, out_channels]
        a_input = torch.cat([x_src, x_dst], dim=-1)  # [E, heads, 2*out_channels]
        # Compute unnormalized attention scores
        alpha = (a_input * self.att).sum(dim=-1)  # [E, heads]
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        # Normalize by target node; softmax over all incoming edges per target
        alpha = softmax(alpha, dst)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Message passing: aggregate neighbor features weighted by attention coefficients
        out = x.new_zeros((N, self.heads, self.out_channels))
        out = out.index_add(0, dst, alpha.unsqueeze(-1) * x_src)
        if self.concat:
            out = out.view(N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        return out + self.bias