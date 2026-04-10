import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, BatchNorm, GlobalAttention, GATv2Conv
from torch_geometric.typing import Adj
from typing import Optional, Tuple, Union
from dataset.data_utils import get_atom_feature_dims, get_bond_feature_dims
from torch_geometric.utils import to_dense_batch, to_dense_adj
import math
from einops import rearrange, repeat, einsum
from torch_scatter import scatter_add

def _max_token_id_from_dims(dims, offset: int = 512) -> int:
    """
    Given a list of per-feature vocabulary sizes (e.g., 9 node feature columns)
    and the fixed offset used in convert_to_single_emb, compute the maximum
    token id that can appear after offsetting.
    """
    feature_num = len(dims)
    max_raw = max(dims) - 1
    max_id = 1 + offset * (feature_num - 1) + max_raw
    return int(max_id)

def _embed_offset_tokens(tokens: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
    """
    tokens: [N, D] long tensor (already offset via convert_to_single_emb)
    emb:    shared nn.Embedding over the whole token space
    Returns: [N, H] dense embedding by summing over the D feature columns.
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(-1)  # [N, 1]
    # [N, D, H] -> [N, H]
    return emb(tokens).sum(dim=1)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1, depth: int = 2):
        super().__init__()
        layers = []
        d_in = in_dim
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(d_in, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d_in = hidden
        layers += [nn.Linear(d_in, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DropPath(nn.Module):
    """
    This implements the path dropping operation in Stochastic Depth.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1) 
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep * mask

class GCNEncoder(nn.Module):
    """
    GNN Encoder:outputs (node_feats h, graph_feats g, batch)
    """
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        offset: int = 512,
    ):
        super().__init__()
        self.hidden = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Consistent with baseline: shared embedding, summed by column
        node_dims = get_atom_feature_dims()
        edge_dims = get_bond_feature_dims()
        node_vocab_size = _max_token_id_from_dims(node_dims, offset=offset) + 1
        edge_vocab_size = _max_token_id_from_dims(edge_dims, offset=offset) + 1

        self.node_emb = nn.Embedding(node_vocab_size, hidden_channels, padding_idx=0)
        self.edge_emb = nn.Embedding(edge_vocab_size, hidden_channels, padding_idx=0)

        # GINEConv stacking + BN/Dropout + residual connections (isomorphic to baseline)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(hidden_channels, hidden_channels, hidden_channels, dropout=dropout, depth=2)
            conv = GINEConv(nn=mlp, train_eps=True)
            self.convs.append(conv)
            self.norms.append(BatchNorm(hidden_channels))
        self.act = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.edge_emb.weight)
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()

    @torch.no_grad()
    def _empty_edge_embed(self, edge_index: Adj, device, dtype):
        return torch.zeros((edge_index.size(1), self.hidden), device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,                 # [N, D_node] long (offset tokens)
        edge_index: Adj,                 # [2, E]
        edge_attr: Optional[torch.Tensor],  # [E, D_edge] long (offset tokens)
        batch: torch.Tensor,             # [N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) Node/edge discrete features → shared embedding with summation by column
        h = _embed_offset_tokens(x, self.node_emb)               # [N, H]
        if edge_attr is not None and edge_attr.numel() > 0:
            e = _embed_offset_tokens(edge_attr, self.edge_emb)   # [E, H]
        else:
            e = self._empty_edge_embed(edge_index, h.device, h.dtype)

        # 2) GINEConv 
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index, e)        # [N, H]
            h = norm(h)
            h = self.act(h)
            h = self.dropout_layer(h)
            h = h + h_in

        # 3) Graph-level readout 
        g = global_add_pool(h, batch)          # [B, H]
        return h, g, batch

class GATEncoder(nn.Module):
    """
    Have the same inputs/outputs with GCNEncoder:
    forward(x, edge_index, edge_attr, batch) -> (h, g, batch)
    """
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        offset: int = 512,
        heads: int = 4,                
        attn_dropout: float = 0.1,
        add_self_loops: bool = False, 
    ):
        super().__init__()
        self.hidden = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_dims = get_atom_feature_dims()
        edge_dims = get_bond_feature_dims()
        node_vocab_size = _max_token_id_from_dims(node_dims, offset=offset) + 1
        edge_vocab_size = _max_token_id_from_dims(edge_dims, offset=offset) + 1
        self.node_emb = nn.Embedding(node_vocab_size, hidden_channels, padding_idx=0)
        self.edge_emb = nn.Embedding(edge_vocab_size, hidden_channels, padding_idx=0)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        out_per_head = hidden_channels // heads
        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_channels,
                out_channels=out_per_head,
                heads=heads,
                edge_dim=hidden_channels,
                dropout=attn_dropout,
                add_self_loops=add_self_loops
            )
            self.convs.append(conv)
            self.norms.append(BatchNorm(hidden_channels))
        self.act = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.edge_emb.weight)
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()

    @torch.no_grad()
    def _empty_edge_embed(self, edge_index: Adj, device, dtype):
        return torch.zeros((edge_index.size(1), self.hidden), device=device, dtype=dtype)

    def forward(self, x, edge_index, edge_attr, batch):
        h = _embed_offset_tokens(x, self.node_emb)               # [N, H]
        if edge_attr is not None and edge_attr.numel() > 0:
            e = _embed_offset_tokens(edge_attr, self.edge_emb)   # [E, H]
        else:
            e = self._empty_edge_embed(edge_index, h.device, h.dtype)

        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index, e)       # [N, H]（heads*out_per_head == hidden）
            h = norm(h)
            h = self.act(h)
            h = self.dropout_layer(h)
            h = h + h_in

        g = global_add_pool(h, batch)        # [B, H]
        return h, g, batch

class MambaBlock(nn.Module):
    """
    Implementation of Mamba block,
    incorporating concepts from state space models and graph neural networks.
    """
    def __init__(self,
                 d_model: int = 128,
                 bias: bool = False,
                 conv_bias: bool = True,
                 d_conv: int = 4,
                 dt_rank: Union[int, str] = 'auto',
                 d_state: int = 2,
                 ):
        super().__init__()

        self.in_proj = nn.Linear(d_model, d_model * 2, bias=bias)
        self.d_model = d_model

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_model,
            padding=d_conv - 1,
        )

        if dt_rank == 'auto':
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)

        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_model)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, dis_dense):

        b, l, d = x.shape

        x_and_res = self.in_proj(x)              # (B, L, 2D)
        x, res = x_and_res.chunk(2, dim=-1)      # (B, L, D), (B, L, D)

        x = rearrange(x, 'b l d -> b d l')       # (B, D, L)
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d l -> b l d')       # (B, L, D)
        x = F.silu(x)

        y = self.ssm(x, dis_dense)               # (B, L, D)
        y = y * F.silu(res)
        out = self.out_proj(y)                   # (B, L, D)
        out[~torch.isfinite(out)] = 0.0
        return out

    def ssm(self, x, dis_dense):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D, dis_dense)

        return y

    def selective_scan(self, u, delta, A, B, C, D, dis_dense):
        """
        Vectorized Graph Selective Scan (no Python for-loop over L)
        Shapes:
        u:       (B, L, D)
        delta:   (B, L, D)
        A:       (D, N)
        B, C:    (B, L, N)
        D:       (D,)
        dis_dense: (B, L, L) or (L, L) or None
        """
        b, L, d_in = u.shape
        n = A.shape[1]

        # MambaBlock.selective_scan
        use_sparse = hasattr(self, "_ctx_cache") and (self._ctx_cache is not None)

        if use_sparse:
            ctx = self._ctx_cache

            self._ctx_cache = None
            node_mask = ctx.node_mask        # (B,L) bool
            edge_index = ctx.edge_index      # (2,E)
            w = ctx.w                        # (E,)
            delta_r = u[node_mask]           # (N,D) 
            row, col = edge_index

            deg = scatter_add(w, row, dim=0, dim_size=delta_r.size(0)).clamp_min(1e-6)
            contrib = (w / deg[row]).unsqueeze(-1) * delta_r[row]
            agg = scatter_add(contrib, col, dim=0, dim_size=delta_r.size(0))   # (N,D)
            delta_p = u.clone()
            delta_p[node_mask] = agg
        elif (dis_dense is not None) and (dis_dense.dim() in (2,3)):
            adj = dis_dense.to(u.dtype)
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(u.size(0), -1, -1)
            delta_p = torch.matmul(adj.transpose(-1, -2), u)
        else:
            delta_p = u

        # Calculate ΔA
        log_deltaA = einsum(delta_p, A, 'b l d, d n -> b l d n').clamp(-20, 20) 
        log_P = torch.cumsum(log_deltaA, dim=1)                                   # (B, L, D, N)
        log_P = log_P.clamp(-30, 30)
        inv_P = torch.exp(-log_P)                                                 # (B, L, D, N)

        # ΔB * u_t
        deltaBu = einsum(delta_p, B, u, 'b l d, b l n, b l d -> b l d n')        # (B, L, D, N)

        # x_t = Σ_{k<=t} (ΔB u_k / Π_{j<=k} ΔA_j) * Π_{j<=t} ΔA_j
        x_all = torch.cumsum(deltaBu * inv_P, dim=1) * torch.exp(log_P)           # (B, L, D, N)

        # y_t = <x_t, C_t>_n + D ⊙ u_t
        y = einsum(x_all, C, 'b l d n, b l n -> b l d') + u * D                  # (B, L, D)
    
        return y




class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class MambaEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 2, drop_path: float = 0.05):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model=d_model) for _ in range(n_layers)])
        self.norms  = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        seq: torch.Tensor,              # [B, L, H] (from to_dense_batch)
        mask: torch.Tensor,             # [B, L] bool
        adj_dense: Optional[torch.Tensor] = None  # [B, L, L] or None
    ):
        x = seq
        for norm, layer in zip(self.norms, self.layers):
            y = layer(norm(x), adj_dense)   
            x = self.drop_path(y) + x

        x = x * mask.unsqueeze(-1)          
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        m = x.sum(dim=1) / denom            
        return m, x                          
    
class GNNMambaFusion(nn.Module):
    """
    The GNN produces atom-level h, 
    which is packed into a sequence and passed to Mamba (optionally with dense adjacency).
    It is then integrated with the graph-level pooled vector g and optional spec_meta to predict spectral graphs.
    """
    def __init__(
        self,
        hidden: int = 128,
        n_bins: int = 4096,
        meta_dim: int = 0,
        use_adj_from_bond: bool = True,  
        n_mamba_layers: int = 2,
        seg_head: bool = False,          
        gnn_type: str = "GCN"
    ):
        super().__init__()
        self.hidden = hidden
        self.n_bins = n_bins
        self.use_adj_from_bond = use_adj_from_bond

        # 1) Encoder
        if gnn_type.lower() == "gat":
            self.gnn_enc = GATEncoder(hidden_channels=hidden)
        else:
            self.gnn_enc = GCNEncoder(hidden_channels=hidden)
            
        self.mamba_enc = MambaEncoder(d_model=hidden, n_layers=n_mamba_layers, drop_path=0.05)

        # 2) GlobalAttention pooling
        self.pools = nn.ModuleList([
            GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
            )) for _ in range(4)
        ])

        # 3) Metadata branch (optional)
        self.meta_mlp = (
            nn.Identity()
            if meta_dim <= 0
            else nn.Sequential(
                nn.Linear(meta_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
        )

        # 4) Head：g(4H) + m(H) + meta(H optional)
        head_in = hidden * 5 + (hidden if meta_dim > 0 else 0)
        self.h_norm = nn.LayerNorm(head_in)

        if not seg_head:
            self.head = nn.Sequential(
                nn.Linear(head_in, 4 * hidden), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(4 * hidden, 4 * hidden), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(4 * hidden, n_bins)
            )
        else:
            K = 8
            seg_bins = (n_bins + K - 1) // K
            self.seg_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(head_in, 2 * hidden), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(2 * hidden, seg_bins)
                ) for _ in range(K)
            ])

    def _build_dense_adj(self, edge_index, batch):
        # Construct dense adjacency from bond edges with parallel normalization
        adj = to_dense_adj(edge_index, batch=batch)             # [B, L, L]
        row_sum = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        adj = adj / row_sum
        return adj

    def forward(self, data):
        # === 1) GNN Encoder：get atom-level h and graph-level g ===
        h, g_add, batch = self.gnn_enc(data.x, data.edge_index, getattr(data, 'edge_attr', None), data.batch)

        # === 2) pooling ===
        g_list = [p(h, batch) for p in self.pools]              # [B, H] × 4
        g = torch.cat(g_list, dim=-1)                           # [B, 4H]

        # === 3) Pack into a sequence and feed in Mamba ===
        seq, mask = to_dense_batch(h, batch)                    # [B, L, H], [B, L]
        adj = None
        if self.use_adj_from_bond:
            adj = self._build_dense_adj(data.edge_index, batch) # [B, L, L]

        m, _ = self.mamba_enc(seq, mask, adj_dense=adj)         # m: [B, H]

        # === 4) Integrate meta and output spectral graph ===
        feats = [g, m]                                          # g: 4H, m: H
        meta = getattr(data, 'spec_meta', None)
        if meta is not None and meta.numel() > 0:
            meta = meta.float()
            if meta.dim() == 3 and meta.size(1) == 1:
                meta = meta.squeeze(1)                          # [B, D]
            feats.append(self.meta_mlp(meta))                   # +H

        h_all = torch.cat(feats, dim=-1)                        # [B, head_in]
        h_all = self.h_norm(h_all)

        if hasattr(self, 'head'):
            out = self.head(h_all)                              # [B, n_bins]
        else:
            outs = [head(h_all) for head in self.seg_heads]
            out = torch.cat(outs, dim=-1)[:, :self.n_bins]

        return out