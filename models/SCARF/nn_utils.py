""" base_hyperopt.py

Abstract away common hyperopt functionality

"""
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Callable

import pytorch_lightning as pl

import ray
from ray import tune
from ray.air.config import RunConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers.async_hyperband import ASHAScheduler

import common


def add_hyperopt_args(parser):
    # Tune args
    ha = parser.add_argument_group("Hyperopt Args")
    ha.add_argument("--cpus-per-trial", default=1, type=int)
    ha.add_argument("--gpus-per-trial", default=1, type=float)
    ha.add_argument("--num-h-samples", default=50, type=int)
    ha.add_argument("--grace-period", default=60 * 15, type=int)
    ha.add_argument("--max-concurrent", default=10, type=int)
    ha.add_argument("--tune-checkpoint", default=None)

    # Overwrite default savedir
    time_name = datetime.now().strftime("%Y_%m_%d")
    save_default = f"results/{time_name}_hyperopt/"
    parser.set_defaults(save_dir=save_default)


def run_hyperopt(
    kwargs: dict,
    score_function: Callable,
    param_space_function: Callable,
    initial_points: list,
):
    """run_hyperopt.

    Args:
        kwargs: All dictionary args for hyperopt and train
        score_function: Trainable function that sets up model train
        param_space_function: Function to suggest new params
        initial_points: List of initial params to try
    """
    ray.init("local")

    # Fix base_args based upon tune args
    kwargs["gpu"] = kwargs.get("gpus_per_trial", 0) > 0
    # max_t = args.max_epochs

    if kwargs["debug"]:
        kwargs["num_h_samples"] = 10
        kwargs["max_epochs"] = 5

    save_dir = kwargs["save_dir"]
    common.setup_logger(
        save_dir, log_name="hyperopt.log", debug=kwargs.get("debug", False)
    )
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Define score function
    trainable = tune.with_parameters(
        score_function, base_args=kwargs, orig_dir=Path().resolve()
    )

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    metric = "val_loss"

    # Include cpus and gpus per trial
    trainable = tune.with_resources(
        trainable,
        resources=tune.PlacementGroupFactory(
            [
                {
                    "CPU": kwargs.get("cpus_per_trial"),
                    "GPU": kwargs.get("gpus_per_trial"),
                },
                {
                    "CPU": kwargs.get("num_workers"),
                },
            ],
            strategy="PACK",
        ),
    )

    search_algo = OptunaSearch(
        metric=metric,
        mode="min",
        points_to_evaluate=initial_points,
        space=param_space_function,
    )
    search_algo = ConcurrencyLimiter(
        search_algo, max_concurrent=kwargs["max_concurrent"]
    )

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            mode="min",
            metric=metric,
            search_alg=search_algo,
            scheduler=ASHAScheduler(
                max_t=24 * 60 * 60,  # max_t,
                time_attr="time_total_s",
                grace_period=kwargs.get("grace_period"),
                reduction_factor=2,
            ),
            num_samples=kwargs.get("num_h_samples"),
        ),
        run_config=RunConfig(name=None, local_dir=kwargs["save_dir"]),
    )

    if kwargs.get("tune_checkpoint") is not None:
        ckpt = str(Path(kwargs["tune_checkpoint"]).resolve())
        tuner = tuner.restore(path=ckpt, restart_errored=True)

    results = tuner.fit()
    best_trial = results.get_best_result()
    output = {"score": best_trial.metrics[metric], "config": best_trial.config}
    out_str = yaml.dump(output, indent=2)
    logging.info(out_str)
    with open(Path(save_dir) / "best_trial.yaml", "w") as f:
        f.write(out_str)

    # Output full res table
    results.get_dataframe().to_csv(
        Path(save_dir) / "full_res_tbl.tsv", sep="\t", index=None
    )

""" dgl_modules.

Directly copy dgl modules to patch them

"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl.function as fn


from dgl.nn import expand_as_pair
import dgl.nn as dgl_nn
import dgl

gcn_msg = fn.copy_u(u="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")


class GatedGraphConv(nn.Module):
    r"""Gated Graph Convolution layer from `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__

    .. math::
        h_{i}^{0} &= [ x_i \| \mathbf{0} ]

        a_{i}^{t} &= \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

        h_{i}^{t+1} &= \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`x_i`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(t+1)}`.
    n_steps : int
        Number of recurrent steps; i.e, the :math:`t` in the above formula.
    n_etypes : int
        Number of edge types.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GatedGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = torch.ones(6, 10)
    >>> conv = GatedGraphConv(10, 10, 2, 3)
    >>> etype = torch.tensor([0,1,2,0,1,2])
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.4652,  0.4458,  0.5169,  0.4126,  0.4847,  0.2303,  0.2757,  0.7721,
            0.0523,  0.0857],
            [ 0.0832,  0.1388, -0.5643,  0.7053, -0.2524, -0.3847,  0.7587,  0.8245,
            0.9315,  0.4063],
            [ 0.6340,  0.4096,  0.7692,  0.2125,  0.2106,  0.4542, -0.0580,  0.3364,
            -0.1376,  0.4948],
            [ 0.5551,  0.7946,  0.6220,  0.8058,  0.5711,  0.3063, -0.5454,  0.2272,
            -0.6931, -0.1607],
            [ 0.2644,  0.2469, -0.6143,  0.6008, -0.1516, -0.3781,  0.5878,  0.7993,
            0.9241,  0.1835],
            [ 0.6393,  0.3447,  0.3893,  0.4279,  0.3342,  0.3809,  0.0406,  0.5030,
            0.1342,  0.0425]], grad_fn=<AddBackward0>)
    """

    def __init__(self, in_feats, out_feats, n_steps, n_etypes, bias=True):
        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        gain = init.calculate_gain("relu")
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, etypes=None):
        """

        Description
        -----------
        Compute Gated Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        etypes : torch.LongTensor, or None
            The edge type tensor of shape :math:`(E,)` where :math:`E` is
            the number of edges of the graph. When there's only one edge type,
            this argument can be skipped

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            assert graph.is_homogeneous, (
                "not a homogeneous graph; convert it with to_homogeneous "
                "and pass in the edge type as argument"
            )
            if self._n_etypes != 1:
                assert (
                    etypes.min() >= 0 and etypes.max() < self._n_etypes
                ), "edge type indices out of range [0, {})".format(self._n_etypes)

            zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
            feat = torch.cat([feat, zero_pad], -1)

            for _ in range(self._n_steps):
                if self._n_etypes == 1 and etypes is None:
                    # Fast path when graph has only one edge type
                    graph.ndata["h"] = self.linears[0](feat)
                    graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "a"))
                    a = graph.ndata.pop("a")  # (N, D)
                else:
                    graph.ndata["h"] = feat
                    for i in range(self._n_etypes):
                        eids = (
                            torch.nonzero(etypes == i, as_tuple=False)
                            .contiguous()
                            .view(-1)
                            .type(graph.idtype)
                        )
                        if len(eids) > 0:
                            graph.apply_edges(
                                lambda edges: {
                                    "W_e*h": self.linears[i](edges.src["h"])
                                },
                                eids,
                            )
                    graph.update_all(fn.copy_e("W_e*h", "m"), fn.sum("m", "a"))
                    a = graph.ndata.pop("a")  # (N, D)
                feat = self.gru(a, feat)
            return feat


def aggregate_mean(h):
    """mean aggregation"""
    return torch.mean(h, dim=1)


def aggregate_max(h):
    """max aggregation"""
    return torch.max(h, dim=1)[0]


def aggregate_min(h):
    """min aggregation"""
    return torch.min(h, dim=1)[0]


def aggregate_sum(h):
    """sum aggregation"""
    return torch.sum(h, dim=1)


def aggregate_std(h):
    """standard deviation aggregation"""
    return torch.sqrt(aggregate_var(h) + 1e-30)


def aggregate_var(h):
    """variance aggregation"""
    h_mean_squares = torch.mean(h * h, dim=1)
    h_mean = torch.mean(h, dim=1)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def _aggregate_moment(h, n):
    """moment aggregation: for each node (E[(X-E[X])^n])^{1/n}"""
    h_mean = torch.mean(h, dim=1, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=1)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + 1e-30, 1.0 / n)
    return rooted_h_n


def aggregate_moment_3(h):
    """moment aggregation with n=3"""
    return _aggregate_moment(h, n=3)


def aggregate_moment_4(h):
    """moment aggregation with n=4"""
    return _aggregate_moment(h, n=4)


def aggregate_moment_5(h):
    """moment aggregation with n=5"""
    return _aggregate_moment(h, n=5)


def scale_identity(h):
    """identity scaling (no scaling operation)"""
    return h


def scale_amplification(h, D, delta):
    """amplification scaling"""
    return h * (np.log(D + 1) / delta)


def scale_attenuation(h, D, delta):
    """attenuation scaling"""
    return h * (delta / np.log(D + 1))


AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": aggregate_moment_3,
    "moment4": aggregate_moment_4,
    "moment5": aggregate_moment_5,
}
SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class PNAConvTower(nn.Module):
    """A single PNA tower in PNA layers"""

    def __init__(
        self,
        in_size,
        out_size,
        aggregators,
        scalers,
        delta,
        dropout=0.0,
        edge_feat_size=0,
    ):
        super(PNAConvTower, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta
        self.edge_feat_size = edge_feat_size

        self.M = nn.Linear(2 * in_size + edge_feat_size, in_size)
        self.U = nn.Linear((len(aggregators) * len(scalers) + 1) * in_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_size)

    def reduce_func(self, nodes):
        """reduce function for PNA layer:
        tensordot of multiple aggregation and scaling operations"""
        msg = nodes.mailbox["msg"]
        degree = msg.size(1)
        h = torch.cat([AGGREGATORS[agg](msg) for agg in self.aggregators], dim=1)
        h = torch.cat(
            [
                SCALERS[scaler](h, D=degree, delta=self.delta)
                if scaler != "identity"
                else h
                for scaler in self.scalers
            ],
            dim=1,
        )
        return {"h_neigh": h}

    def message(self, edges):
        """message function for PNA layer"""
        if self.edge_feat_size > 0:
            f = torch.cat([edges.src["h"], edges.dst["h"], edges.data["a"]], dim=-1)
        else:
            f = torch.cat([edges.src["h"], edges.dst["h"]], dim=-1)
        return {"msg": self.M(f)}

    def forward(self, graph, node_feat, edge_feat=None):
        """compute the forward pass of a single tower in PNA convolution layer"""
        # calculate graph normalization factors
        snorm_n = torch.cat(
            [torch.ones(N, 1).to(node_feat) / N for N in graph.batch_num_nodes()], dim=0
        ).sqrt()
        with graph.local_scope():
            graph.ndata["h"] = node_feat
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat

            graph.update_all(self.message, self.reduce_func)
            h = self.U(torch.cat([node_feat, graph.ndata["h_neigh"]], dim=-1))
            h = h * snorm_n
            return self.dropout(self.batchnorm(h))


class PNAConv(nn.Module):
    r"""Principal Neighbourhood Aggregation Layer from `Principal Neighbourhood Aggregation
    for Graph Nets <https://arxiv.org/abs/2004.05718>`__

    A PNA layer is composed of multiple PNA towers. Each tower takes as input a split of the
    input features, and computes the message passing as below.

    .. math::
        h_i^(l+1) = U(h_i^l, \oplus_{(i,j)\in E}M(h_i^l, e_{i,j}, h_j^l))

    where :math:`h_i` and :math:`e_{i,j}` are node features and edge features, respectively.
    :math:`M` and :math:`U` are MLPs, taking the concatenation of input for computing
    output features. :math:`\oplus` represents the combination of various aggregators
    and scalers. Aggregators aggregate messages from neighbours and scalers scale the
    aggregated messages in different ways. :math:`\oplus` concatenates the output features
    of each combination.

    The output of multiple towers are concatenated and fed into a linear mixing layer for the
    final output.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    aggregators : list of str
        List of aggregation function names(each aggregator specifies a way to aggregate
        messages from neighbours), selected from:

        * ``mean``: the mean of neighbour messages

        * ``max``: the maximum of neighbour messages

        * ``min``: the minimum of neighbour messages

        * ``std``: the standard deviation of neighbour messages

        * ``var``: the variance of neighbour messages

        * ``sum``: the sum of neighbour messages

        * ``moment3``, ``moment4``, ``moment5``: the normalized moments aggregation
        :math:`(E[(X-E[X])^n])^{1/n}`
    scalers: list of str
        List of scaler function names, selected from:

        * ``identity``: no scaling

        * ``amplification``: multiply the aggregated message by :math:`\log(d+1)/\delta`,
        where :math:`d` is the degree of the node.

        * ``attenuation``: multiply the aggregated message by :math:`\delta/\log(d+1)`
    delta: float
        The degree-related normalization factor computed over the training set, used by scalers
        for normalization. :math:`E[\log(d+1)]`, where :math:`d` is the degree for each node
        in the training set.
    dropout: float, optional
        The dropout ratio. Default: 0.0.
    num_towers: int, optional
        The number of towers used. Default: 1. Note that in_size and out_size must be divisible
        by num_towers.
    edge_feat_size: int, optional
        The edge feature size. Default: 0.
    residual : bool, optional
        The bool flag that determines whether to add a residual connection for the
        output. Default: True. If in_size and out_size of the PNA conv layer are not
        the same, this flag will be set as False forcibly.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import PNAConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = torch.ones(6, 10)
    >>> conv = PNAConv(10, 10, ['mean', 'max', 'sum'], ['identity', 'amplification'], 2.5)
    >>> ret = conv(g, feat)
    """

    def __init__(
        self,
        in_size,
        out_size,
        aggregators,
        scalers,
        delta,
        dropout=0.0,
        num_towers=1,
        edge_feat_size=0,
        residual=True,
    ):
        super(PNAConv, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        assert in_size % num_towers == 0, "in_size must be divisible by num_towers"
        assert out_size % num_towers == 0, "out_size must be divisible by num_towers"
        self.tower_in_size = in_size // num_towers
        self.tower_out_size = out_size // num_towers
        self.edge_feat_size = edge_feat_size
        self.residual = residual
        if self.in_size != self.out_size:
            self.residual = False

        self.towers = nn.ModuleList(
            [
                PNAConvTower(
                    self.tower_in_size,
                    self.tower_out_size,
                    aggregators,
                    scalers,
                    delta,
                    dropout=dropout,
                    edge_feat_size=edge_feat_size,
                )
                for _ in range(num_towers)
            ]
        )

        self.mixing_layer = nn.Sequential(nn.Linear(out_size, out_size), nn.LeakyReLU())

    def forward(self, graph, node_feat, edge_feat=None):
        r"""
        Description
        -----------
        Compute PNA layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            should be the same as out_size.
        """
        h_cat = torch.cat(
            [
                tower(
                    graph,
                    node_feat[
                        :, ti * self.tower_in_size : (ti + 1) * self.tower_in_size
                    ],
                    edge_feat,
                )
                for ti, tower in enumerate(self.towers)
            ],
            dim=1,
        )
        h_out = self.mixing_layer(h_cat)
        # add residual connection
        if self.residual:
            h_out = h_out + node_feat

        return h_out


class GINEConv(nn.Module):
    r"""Graph Isomorphism Network with Edge Features, introduced by
    `Strategies for Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \sum_{j\in\mathcal{N}(i)}\mathrm{ReLU}(h_j^{l} + e_{j,i}^{l})\right)

    where :math:`e_{j,i}^{l}` is the edge feature.

    Parameters
    ----------
    apply_func : callable module or None
        The :math:`f_\Theta` in the formula. If not None, it will be applied to
        the updated node features. The default value is None.
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.

    Examples
    --------

    >>> import dgl
    >>> import torch
    >>> import torch.nn as nn
    >>> from dgl.nn import GINEConv

    >>> g = dgl.graph(([0, 1, 2], [1, 1, 3]))
    >>> in_feats = 10
    >>> out_feats = 20
    >>> nfeat = torch.randn(g.num_nodes(), in_feats)
    >>> efeat = torch.randn(g.num_edges(), in_feats)
    >>> conv = GINEConv(nn.Linear(in_feats, out_feats))
    >>> res = conv(g, nfeat, efeat)
    >>> print(res.shape)
    torch.Size([4, 20])
    """

    def __init__(self, apply_func=None, init_eps=0, learn_eps=False):
        super(GINEConv, self).__init__()
        self.apply_func = apply_func
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def message(self, edges):
        r"""User-defined Message Function"""
        return {"m": F.relu(edges.src["hn"] + edges.data["he"])}

    def forward(self, graph, node_feat, edge_feat):
        r"""Forward computation.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it is the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input feature size requirement of ``apply_func``.
        edge_feat : torch.Tensor
            Edge feature. It is a tensor of shape :math:`(E, D_{in})` where :math:`E`
            is the number of edges.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output feature size of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as :math:`D_{in}`.
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_feat, graph)
            graph.srcdata["hn"] = feat_src
            graph.edata["he"] = edge_feat
            graph.update_all(self.message, fn.sum("m", "neigh"))
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata["h"] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata["h"]
            # return h
            return self.linear(h)

class MultiEdgeGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, etypes):
        super().__init__()
        self.etype_layers = nn.ModuleDict({
            etype: dgl_nn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True)
            for etype in etypes
        })

    def forward(self, g, inputs):
        # Input: node features (same for all nodes, since there's a single node type)
        g.ndata['h'] = inputs

        outputs = torch.zeros_like(inputs)

        with g.local_scope():
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                rel_graph.update_all(
                    dgl.function.copy_u('h', 'm'),
                    dgl.function.sum('m', 'h_agg'),
                    etype=etype
                )
                outputs += self.etype_layers[etype](rel_graph, rel_graph.ndata['h_agg'])
        
        return outputs
    
class MultiEdgeGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, etypes, conv_steps):
        super().__init__()
        self.layer1 = MultiEdgeGCNLayer(in_feats, hidden_feats, etypes)
        self.layer2 = MultiEdgeGCNLayer(hidden_feats, out_feats, etypes)
        self.conv_steps = conv_steps

    def forward(self, g, inputs):
        h = inputs
        for _ in range(self.conv_steps-1):
            h = self.layer1(g, h)
            h = F.relu(h)
        h = self.layer2(g, h)
        return h

class HyperGNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_conv, dropout=0):
        super(HyperGNN, self).__init__()
        self.layer = GCNLayer(hidden_size, hidden_size)
        self.layer_out = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.dropout_conv = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)

        self.num_conv = num_conv

    def forward(self, g, features):
        for _ in range(self.num_conv):
            features = self.layer(g, features)
            features = self.activation(features)
            features = self.dropout_conv(features)

        result=self.layer_out(features)
        result=self.dropout_output(result)

        return result
import torch
import torch.nn as nn
import numpy as np

import common


class IntFeaturizer(nn.Module):
    """
    Base class for mapping integers to a vector representation (primarily to be used as a "richer" embedding for NNs
    processing integers).

    Subclasses should define `self.int_to_feat_matrix`, a matrix where each row is the vector representation for that
    integer, i.e. to get a vector representation for `5`, one could call `self.int_to_feat_matrix[5]`.

    Note that this class takes care of creating a fixed number (`self.NUM_EXTRA_EMBEDDINGS` to be precise) of extra
    "learned" embeddings these will be concatenated after the integer embeddings in the forward pass,
    be learned, and be used for extra  non-integer tokens such as the "to be confirmed token" (i.e., pad) token.
    They are indexed starting from `self.MAX_COUNT_INT`.
    """

    MAX_COUNT_INT = 255  # the maximum number of integers that we are going to see as a "count", i.e. 0 to MAX_COUNT_INT-1
    NUM_EXTRA_EMBEDDINGS = 1  # Number of extra embeddings to learn -- one for the "to be confirmed" embedding.

    def __init__(self, embedding_dim):
        super().__init__()
        weights = torch.zeros(self.NUM_EXTRA_EMBEDDINGS, embedding_dim)
        self._extra_embeddings = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self._extra_embeddings, 0.0, 1.0)
        self.embedding_dim = embedding_dim

    def forward(self, tensor):
        """
        Convert the integer `tensor` into its new representation -- note that it gets stacked along final dimension.
        """
        orig_shape = tensor.shape
        out_tensor = torch.empty(
            (*orig_shape, self.embedding_dim), device=tensor.device
        )
        extra_embed = tensor >= self.MAX_COUNT_INT

        tensor = tensor.long()
        norm_embeds = self.int_to_feat_matrix[tensor[~extra_embed]]
        extra_embeds = self._extra_embeddings[tensor[extra_embed] - self.MAX_COUNT_INT]

        out_tensor[~extra_embed] = norm_embeds
        out_tensor[extra_embed] = extra_embeds

        temp_out = out_tensor.reshape(*orig_shape[:-1], -1)
        return temp_out

    @property
    def num_dim(self):
        return self.int_to_feat_matrix.shape[1]

    @property
    def full_dim(self):
        return self.num_dim * common.NORM_VEC.shape[0]


class FourierFeaturizer(IntFeaturizer):
    """
    Inspired by:
    Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
    Barron, J.T. and Ng, R. (2020) ‘Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
     Domains’, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

    Some notes:
    * we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
        Binarizer quite closely but be a bit smoother.
    """

    def __init__(self):

        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2
        # ^ need at least this many to ensure that the whole input range can be represented on the half circle.

        freqs = 0.5 ** torch.arange(num_freqs, dtype=torch.float32)
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(
            embedding_dim=2 * freqs_time_2pi.shape[0]
        )  # 2 for cosine and sine

        # we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
        combo_of_sinusoid_args = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        all_features = torch.cat(
            [torch.cos(combo_of_sinusoid_args), torch.sin(combo_of_sinusoid_args)],
            dim=1,
        )

        # ^ shape:  MAX_COUNT_INT x 2 * num_freqs
        self.int_to_feat_matrix = nn.Parameter(all_features.float())
        self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerSines(IntFeaturizer):
    """
    Like other fourier feats but sines only

    Inspired by:
    Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
    Barron, J.T. and Ng, R. (2020) ‘Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
     Domains’, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

    Some notes:
    * we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
        Binarizer quite closely but be a bit smoother.
    """

    def __init__(self):

        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2
        # ^ need at least this many to ensure that the whole input range can be represented on the half circle.

        freqs = (0.5 ** torch.arange(num_freqs, dtype=torch.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(embedding_dim=freqs_time_2pi.shape[0])

        # we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
        combo_of_sinusoid_args = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        # ^ shape:  MAX_COUNT_INT x 2 * num_freqs
        self.int_to_feat_matrix = nn.Parameter(
            torch.sin(combo_of_sinusoid_args).float()
        )
        self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerAbsoluteSines(IntFeaturizer):
    """
    Like other fourier feats but sines only and absoluted.

    Inspired by:
    Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R.,
    Barron, J.T. and Ng, R. (2020) ‘Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
     Domains’, arXiv [cs.CV]. Available at: http://arxiv.org/abs/2006.10739.

    Some notes:
    * we'll put the frequencies at powers of 1/2 rather than random Gaussian samples; this means it will match the
        Binarizer quite closely but be a bit smoother.
    """

    def __init__(self):

        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2

        freqs = (0.5 ** torch.arange(num_freqs, dtype=torch.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs

        super().__init__(embedding_dim=freqs_time_2pi.shape[0])

        # we will define the features at this frequency up front (as we only will ever see a fixed number of counts):
        combo_of_sinusoid_args = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        # ^ shape:  MAX_COUNT_INT x 2 * num_freqs
        self.int_to_feat_matrix = nn.Parameter(
            torch.abs(torch.sin(combo_of_sinusoid_args)).float()
        )
        self.int_to_feat_matrix.requires_grad = False


class RBFFeaturizer(IntFeaturizer):
    """
    A featurizer that puts radial basis functions evenly between 0 and max_count-1. These will have a width of
    (max_count-1) / (num_funcs) to decay to about 0.6 of its original height at reaching the next func.

    """

    def __init__(self, num_funcs=32):
        """
        :param num_funcs: number of radial basis functions to use: their width will automatically be chosen -- see class
                            docstring.
        """
        super().__init__(embedding_dim=num_funcs)
        width = (self.MAX_COUNT_INT - 1) / num_funcs
        centers = torch.linspace(0, self.MAX_COUNT_INT - 1, num_funcs)

        pre_exponential_terms = (
            -0.5
            * ((torch.arange(self.MAX_COUNT_INT)[:, None] - centers[None, :]) / width)
            ** 2
        )
        # ^ shape: MAX_COUNT_INT x num_funcs
        feats = torch.exp(pre_exponential_terms)

        self.int_to_feat_matrix = nn.Parameter(feats.float())
        self.int_to_feat_matrix.requires_grad = False


class OneHotFeaturizer(IntFeaturizer):
    """
    A featurizer that turns integers into their one hot encoding.

    Represents:
     - 0 as 1000000000...
     - 1 as 0100000000...
     - 2 as 0010000000...
     and so on.
    """

    def __init__(self):
        super().__init__(embedding_dim=self.MAX_COUNT_INT)
        feats = torch.eye(self.MAX_COUNT_INT)
        self.int_to_feat_matrix = nn.Parameter(feats.float())
        self.int_to_feat_matrix.requires_grad = False


class LearnedFeaturizer(IntFeaturizer):
    """
    Learns the features for the different integers.

    Pretty much `nn.Embedding` but we get to use the forward of the superclass which behaves a bit differently.
    """

    def __init__(self, feature_dim=32):
        super().__init__(embedding_dim=feature_dim)
        weights = torch.zeros(self.MAX_COUNT_INT, feature_dim)
        self.int_to_feat_matrix = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self.int_to_feat_matrix, 0.0, 1.0)


class FloatFeaturizer(IntFeaturizer):
    """
    Norms the features
    """

    def __init__(self):
        # Norm vec
        # Placeholder..
        super().__init__(embedding_dim=1)
        self.norm_vec = torch.from_numpy(common.NORM_VEC).float()
        self.norm_vec = nn.Parameter(self.norm_vec)
        self.norm_vec.requires_grad = False

    def forward(self, tensor):
        """
        Convert the integer `tensor` into its new representation -- note that it gets stacked along final dimension.
        """
        tens_shape = tensor.shape
        out_shape = [1] * (len(tens_shape) - 1) + [-1]
        return tensor / self.norm_vec.reshape(*out_shape)

    @property
    def num_dim(self):
        return 1


def get_embedder(embedder):
    if embedder == "fourier":
        embedder = FourierFeaturizer()
    elif embedder == "rbf":
        embedder = RBFFeaturizer()
    elif embedder == "one-hot":
        embedder = OneHotFeaturizer()
    elif embedder == "learnt":
        embedder = LearnedFeaturizer()
    elif embedder == "float":
        embedder = FloatFeaturizer()
    elif embedder == "fourier-sines":
        embedder = FourierFeaturizerSines()
    elif embedder == "abs-sines":
        embedder = FourierFeaturizerAbsoluteSines()
    else:
        raise NotImplementedError
    return embedder

""" mol_graph.py.

Classes to featurize molecules into a graph with onehot concat feats on atoms
and bonds. Inspired by the dgllife library.

"""
from rdkit import Chem
import numpy as np
import torch
import dgl
atom_feat_registry = {}
bond_feat_registry = {}


def register_bond_feat(cls):
    """register_bond_feat."""
    bond_feat_registry[cls.name] = {"fn": cls.featurize, "feat_size": cls.feat_size}
    return cls


def register_atom_feat(cls):
    """register_atom_feat."""
    atom_feat_registry[cls.name] = {"fn": cls.featurize, "feat_size": cls.feat_size}
    return cls


class MolDGLGraph:
    def __init__(
        self,
        atom_feats: list = [
            "a_onehot",
            "a_degree",
            "a_hybrid",
            "a_formal",
            "a_radical",
            "a_ring",
            "a_mass",
            "a_chiral",
        ],
        bond_feats: list = ["b_degree"],
        pe_embed_k: int = 0,
    ):
        """__init__

        Args:
            atom_feats (list)
            bond_feats (list)
            pe_embed_k (int)

        """
        self.pe_embed_k = pe_embed_k
        self.atom_feats = atom_feats
        self.bond_feats = bond_feats
        self.a_featurizers = []
        self.b_featurizers = []

        self.num_atom_feats = 0
        self.num_bond_feats = 0

        for i in self.atom_feats:
            if i not in atom_feat_registry:
                raise ValueError(f"Feat {i} not recognized")
            feat_obj = atom_feat_registry[i]
            self.num_atom_feats += feat_obj["feat_size"]
            self.a_featurizers.append(feat_obj["fn"])

        for i in self.bond_feats:
            if i not in bond_feat_registry:
                raise ValueError(f"Feat {i} not recognized")
            feat_obj = bond_feat_registry[i]
            self.num_bond_feats += feat_obj["feat_size"]
            self.b_featurizers.append(feat_obj["fn"])

        self.num_atom_feats += self.pe_embed_k

    def get_mol_graph(
        self,
        mol: Chem.Mol,
        bigraph: str = True,
    ) -> dict:
        """get_mol_graph.

        Args:
            mol (Chem.Mol):
            bigraph (bool): If true, double all edges.

        Return:
            dict:
                "atom_feats": np.ndarray (|N| x d_n)
                "bond_feats": np.ndarray (|E| x d_e)
                "bond_tuples": np.ndarray (|E| x 2)

        """
        all_atoms = mol.GetAtoms()
        all_bonds = mol.GetBonds()
        bond_feats = []
        bond_tuples = []
        atom_feats = []
        for bond in all_bonds:
            strt = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_tuples.append((strt, end))
            bond_feat = []
            for fn in self.b_featurizers:
                bond_feat.extend(fn(bond))
            bond_feats.append(bond_feat)

        for atom in all_atoms:
            atom_feat = []
            for fn in self.a_featurizers:
                atom_feat.extend(fn(atom))
            atom_feats.append(atom_feat)

        atom_feats = np.array(atom_feats)
        bond_feats = np.array(bond_feats)
        bond_tuples = np.array(bond_tuples)

        # Add doubles
        if bigraph:
            rev_bonds = np.vstack([bond_tuples[:, 1], bond_tuples[:, 0]]).transpose(
                1, 0
            )
            bond_tuples = np.vstack([bond_tuples, rev_bonds])
            bond_feats = np.vstack([bond_feats, bond_feats])
        return {
            "atom_feats": atom_feats,
            "bond_feats": bond_feats,
            "bond_tuples": bond_tuples,
        }

    def get_dgl_graph(self, mol: Chem.Mol, bigraph: str = True):
        """get_dgl_graph.

        Args:
            mol (Chem.Mol):
            bigraph (bool): If true, double all edges.

        Return:
            dgl graph object
        """
        mol_graph = self.get_mol_graph(mol, bigraph=bigraph)

        bond_inds = torch.from_numpy(mol_graph["bond_tuples"]).long()
        bond_feats = torch.from_numpy(mol_graph["bond_feats"]).float()
        atom_feats = torch.from_numpy(mol_graph["atom_feats"]).float()

        g = dgl.graph(
            data=(bond_inds[:, 0], bond_inds[:, 1]), num_nodes=atom_feats.shape[0]
        )
        g.ndata["h"] = atom_feats
        g.edata["e"] = bond_feats

        if self.pe_embed_k > 0:
            pe_embeds = random_walk_pe(
                g,
                k=self.pe_embed_k,
            )
            g.ndata["h"] = torch.cat((g.ndata["h"], pe_embeds), -1)

        return g


class FeatBase:
    """FeatBase.

    Extend this class for atom and bond featurizers

    """

    feat_size = 0
    name = "base"

    @classmethod
    def featurize(cls, x) -> list:
        raise NotImplementedError()


@register_atom_feat
class AtomOneHot(FeatBase):
    name = "a_onehot"
    allowable_set = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "As",
        "Al",
        "I",
        "B",
        "V",
        "K",
        "Tl",
        "Yb",
        "Sb",
        "Sn",
        "Ag",
        "Pd",
        "Co",
        "Se",
        "Ti",
        "Zn",
        "H",
        "Li",
        "Ge",
        "Cu",
        "Au",
        "Ni",
        "Cd",
        "In",
        "Mn",
        "Zr",
        "Cr",
        "Pt",
        "Hg",
        "Pb",
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        return one_hot_encoding(x.GetSymbol(), cls.allowable_set, True)


@register_atom_feat
class AtomDegree(FeatBase):
    name = "a_degree"
    allowable_set = list(range(11))
    feat_size = len(allowable_set) + 1 + 2

    @classmethod
    def featurize(cls, x) -> int:
        deg = [x.GetDegree(), x.GetTotalDegree()]
        onehot = one_hot_encoding(deg, cls.allowable_set, True)
        return deg + onehot


@register_atom_feat
class AtomHybrid(FeatBase):

    name = "a_hybrid"
    allowable_set = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        onehot = one_hot_encoding(x.GetHybridization(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomFormal(FeatBase):

    name = "a_formal"
    allowable_set = list(range(-2, 3))
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        onehot = one_hot_encoding(x.GetFormalCharge(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomRadical(FeatBase):

    name = "a_radical"
    allowable_set = list(range(5))
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        onehot = one_hot_encoding(x.GetNumRadicalElectrons(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomRing(FeatBase):

    name = "a_ring"
    allowable_set = [True, False]
    feat_size = len(allowable_set) * 2

    @classmethod
    def featurize(cls, x) -> int:
        onehot_ring = one_hot_encoding(x.IsInRing(), cls.allowable_set, False)
        onehot_aromatic = one_hot_encoding(x.GetIsAromatic(), cls.allowable_set, False)
        return onehot_ring + onehot_aromatic


@register_atom_feat
class AtomChiral(FeatBase):

    name = "a_chiral"
    allowable_set = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        chiral_onehot = one_hot_encoding(x.GetChiralTag(), cls.allowable_set, True)
        return chiral_onehot


@register_atom_feat
class AtomMass(FeatBase):

    name = "a_mass"
    coef = 0.01
    feat_size = 1

    @classmethod
    def featurize(cls, x) -> int:
        return [x.GetMass() * cls.coef]


@register_bond_feat
class BondDegree(FeatBase):

    name = "b_degree"
    allowable_set = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        return one_hot_encoding(x.GetBondType(), cls.allowable_set, True)


@register_bond_feat
class BondStereo(FeatBase):

    name = "b_stereo"
    allowable_set = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x) -> int:
        return one_hot_encoding(x.GetStereo(), cls.allowable_set, True)


@register_bond_feat
class BondConj(FeatBase):

    name = "b_ring"
    feat_size = 2

    @classmethod
    def featurize(cls, x) -> int:
        return one_hot_encoding(x.IsInRing(), [False, True], False)


@register_bond_feat
class BondConj(FeatBase):

    name = "b_conj"
    feat_size = 2

    @classmethod
    def featurize(cls, x) -> int:
        return one_hot_encoding(x.GetIsConjugated(), [False, True], False)


def one_hot_encoding(x, allowable_set, encode_unknown=False) -> list:
    """One_hot encoding.

    Code taken from dgllife library
    https://lifesci.dgl.ai/_modules/dgllife/utils/featurizers.html

    Args:
        x: Val to encode
        allowable_set: Options
        encode_unknown: If true, encode unk

    Return:
        list of bools
    """

    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: int(x == s), allowable_set))

""" nn_utils.py
"""
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import dgl
from packaging.version import Version

if Version(torch.__version__) > Version('2.0.0'):
    _TORCH_SP_SUPPORT = True  # use torch built-in sparse
else:
    try:
        import torch_sparse
        _TORCH_SP_SUPPORT = False  # use torch_sparse package
    except:
        raise ModuleNotFoundError("Please either install torch_sparse or upgrade to a PyTorch version that supports "
                                  "sparse-sparse matrix multiply")

from dgl.backend import pytorch as dgl_F


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_lr_scheduler(
    optimizer, lr_decay_rate: float, decay_steps: int = 5000, warmup: int = 1000
):
    """build_lr_scheduler.

    Args:
        optimizer:
        lr_decay_rate (float): lr_decay_rate
        decay_steps (int): decay_steps
        warmup_steps (int): warmup_steps
    """

    def lr_lambda(step):
        if step >= warmup:
            # Adjust
            step = step - warmup
            rate = lr_decay_rate ** (step // decay_steps)
        else:
            rate = 1 - math.exp(-step / warmup)
        return rate

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


class MoleculeGNN(nn.Module):
    """MoleculeGNN Module"""

    def __init__(
        self,
        hidden_size: int,
        num_step_message_passing: int = 4,
        gnn_node_feats: int = 74,
        gnn_edge_feats: int = 4,  # 12,
        mpnn_type: str = "GGNN",
        node_feat_symbol="h",
        set_transform_layers: int = 2,
        dropout: float = 0,
        **kwargs
    ):
        """__init__.
        Args:
            hidden_size (int): Hidden size
            num_mol_layers (int): Number of layers to encode for the molecule
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn_edge_feats = gnn_edge_feats
        self.gnn_node_feats = gnn_node_feats
        self.node_feat_symbol = node_feat_symbol
        self.dropout = dropout

        self.mpnn_type = mpnn_type
        self.hidden_size = hidden_size
        self.num_step_message_passing = num_step_message_passing
        self.input_project = nn.Linear(self.gnn_node_feats, self.hidden_size)

        if self.mpnn_type == "GGNN":
            self.gnn = GGNN(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
            )
        elif self.mpnn_type == "PNA":
            self.gnn = PNA(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
                dropout=self.dropout,
            )
        elif self.mpnn_type == "GINE":
            self.gnn = GINE(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
                dropout=self.dropout,
            )
        else:
            raise ValueError()

        # Keeping d_head only to 2x increase in size to avoid memory. Orig
        # transformer uses 4x
        self.set_transformer = SetTransformerEncoder(
            d_model=self.hidden_size,
            n_heads=4,
            d_head=self.hidden_size // 4,
            d_ff=hidden_size,
            n_layers=set_transform_layers,
        )

    def forward(self, g):
        """encode batch of molecule graph"""
        with g.local_scope():
            # Set initial hidden
            ndata = g.ndata[self.node_feat_symbol]
            edata = g.edata["e"]
            h_init = self.input_project(ndata)
            g.ndata.update({"_h": h_init})
            g.edata.update({"_e": edata})

            if self.mpnn_type == "GGNN":
                # Get graph output
                output = self.gnn(g, "_h", "_e")
            elif self.mpnn_type == "PNA":
                # Get graph output
                output = self.gnn(g, "_h", "_e")
            elif self.mpnn_type == "GINE":
                # Get graph output
                output = self.gnn(g, "_h", "_e")
            else:
                raise NotImplementedError()

        output = self.set_transformer(g, output)
        return output


class GINE(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        edge_feats=4,
        num_step_message_passing=4,
        dropout=0,
        **kwargs
    ):
        """GINE.

        Args:
            input_size (int): Size of edge features into the graph
            hidden_size (int): Hidden size
            edge_feats (int): Number of edge feats. Must be onehot!
            node_feats (int): Num of node feats (default 74)
            num_step_message_passing (int): Number of message passing steps
            dropout
        """
        super().__init__()

        self.edge_transform = nn.Linear(edge_feats, hidden_size)

        self.layers = []
        for i in range(num_step_message_passing):
            apply_fn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            temp_layer = GINEConv(apply_func=apply_fn, init_eps=0)
            self.layers.append(temp_layer)

        self.layers = nn.ModuleList(self.layers)
        self.bnorms = get_clones(nn.BatchNorm1d(hidden_size), num_step_message_passing)
        self.dropouts = get_clones(nn.Dropout(dropout), num_step_message_passing)

    def forward(self, graph, nfeat_name="_h", efeat_name="_e"):
        """forward.

        Args:
            graph (dgl graph): Graph object
            nfeat_name (str): Name of node feat data
            efeat_name (str): Name of e feat

        Return:
            h: Hidden state at each node


        """
        node_feat, edge_feat = graph.ndata[nfeat_name], graph.edata[efeat_name]
        edge_feat = self.edge_transform(edge_feat)

        for dropout, layer, norm in zip(self.dropouts, self.layers, self.bnorms):
            layer_out = layer(graph, node_feat, edge_feat)
            node_feat = F.relu(dropout(norm(layer_out))) + node_feat

        return node_feat


class GGNN(nn.Module):
    def __init__(
        self, hidden_size=64, edge_feats=4, num_step_message_passing=4, **kwargs
    ):
        """GGNN.

        Define a gated graph neural network

        This is very similar to the NNConv models.

        Args:
            input_size (int): Size of edge features into the graph
            hidden_size (int): Hidden size
            edge_feats (int): Number of edge feats. Must be onehot!
            node_feats (int): Num of node feats (default 74)
            num_step_message_passing (int): Number of message passing steps
        """
        super().__init__()
        self.model = GatedGraphConv(
            in_feats=hidden_size,
            out_feats=hidden_size,
            n_steps=num_step_message_passing,
            n_etypes=edge_feats,
        )

    def forward(self, graph, nfeat_name="_h", efeat_name="_e"):
        """forward.

        Args:
            graph (dgl graph): Graph object
            nfeat_name (str): Name of node feat data
            efeat_name (str): Name of e feat

        Return:
            h: Hidden state at each node

        """
        if "e_ind" in graph.edata:
            etypes = graph.edata["e_ind"]
        else:
            etypes = graph.edata[efeat_name].argmax(1)
        return self.model(graph, graph.ndata[nfeat_name], etypes=etypes)


class PNA(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        edge_feats=4,
        num_step_message_passing=4,
        dropout=0,
        **kwargs
    ):
        """PNA.

        Define a PNA network

        Args:
            input_size (int): Size of edge features into the graph
            hidden_size (int): Hidden size
            edge_feats (int): Number of edge feats. Must be onehot!
            node_feats (int): Num of node feats (default 74)
            num_step_message_passing (int): Number of message passing steps
        """
        super().__init__()
        self.layer = PNAConv(
            in_size=hidden_size,
            out_size=hidden_size,
            aggregators=["mean", "max", "min", "std", "var", "sum"],
            scalers=["identity", "amplification", "attenuation"],
            delta=2.5,
            dropout=dropout,
        )

        self.layers = get_clones(self.layer, num_step_message_passing)
        self.bnorms = get_clones(nn.BatchNorm1d(hidden_size), num_step_message_passing)

    def forward(self, graph, nfeat_name="_h", efeat_name="_e"):
        """forward.

        Args:
            graph (dgl graph): Graph object
            nfeat_name (str): Name of node feat data
            efeat_name (str): Name of e feat

        Return:
            h: Hidden state at each node

        """
        node_feat, edge_feat = graph.ndata[nfeat_name], graph.edata[efeat_name]
        for layer, norm in zip(self.layers, self.bnorms):
            node_feat = F.relu(norm(layer(graph, node_feat, edge_feat))) + node_feat

        return node_feat


class MLPBlocks(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        output_size: int = None,
        use_residuals: bool = False,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = get_clones(middle_layer, num_layers - 1)

        self.output_layer = None
        self.output_size = output_size
        if self.output_size is not None:
            self.output_layer = nn.Linear(hidden_size, self.output_size)

        self.use_residuals = use_residuals
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn_input = nn.BatchNorm1d(hidden_size)
            bn = nn.BatchNorm1d(hidden_size)
            self.bn_mids = get_clones(bn, num_layers - 1)

    def safe_apply_bn(self, x, bn):
        """transpose and untranspose after linear for 3 dim items to us
        batchnorm"""
        temp_shape = x.shape
        if len(x.shape) == 2:
            return bn(x)
        elif len(x.shape) == 3:
            return bn(x.transpose(-1, -2)).transpose(-1, -2)
        else:
            raise NotImplementedError()

    def forward(self, x):
        output = x
        output = self.input_layer(x)
        output = self.activation(output)
        output = self.dropout_layer(output)

        if self.use_batchnorm:
            output = self.safe_apply_bn(output, self.bn_input)

        old_op = output
        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.activation(output)
            output = self.dropout_layer(output)

            if self.use_batchnorm:
                output = self.safe_apply_bn(output, self.bn_mids[layer_index])

            if self.use_residuals:
                output += old_op
                old_op = output

        if self.output_layer is not None:
            output = self.output_layer(output)

        return output


# DGL Models
# https://docs.dgl.ai/en/0.6.x/_modules/dgl/nn/pytorch/glob.html#SetTransformerDecoder


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention block, used in Transformer, Set Transformer and so on.

    Parameters
    ----------
    d_model : int
        The feature size (input and output) in Multi-Head Attention layer.
    num_heads : int
        The number of heads.
    d_head : int
        The hidden size per head.
    d_ff : int
        The inner hidden size in the Feed-Forward Neural Network.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Notes
    -----
    This module was used in SetTransformer layer.
    """

    def __init__(self, d_model, num_heads, d_head, d_ff, dropouth=0.0, dropouta=0.0):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.proj_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.proj_o = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropouth),
            nn.Linear(d_ff, d_model),
        )
        self.droph = nn.Dropout(dropouth)
        self.dropa = nn.Dropout(dropouta)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_inter = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def self_attention(self, x, mem, lengths_x, lengths_mem):
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        max_len_mem = max(lengths_mem)
        device = x.device

        lengths_x = lengths_x.clone().detach().long().to(device)
        lengths_mem = lengths_mem.clone().detach().long().to(device)

        queries = self.proj_q(x).view(-1, self.num_heads, self.d_head)
        keys = self.proj_k(mem).view(-1, self.num_heads, self.d_head)
        values = self.proj_v(mem).view(-1, self.num_heads, self.d_head)

        # padding to (B, max_len_x/mem, num_heads, d_head)
        queries = dgl_F.pad_packed_tensor(queries, lengths_x, 0)
        keys = dgl_F.pad_packed_tensor(keys, lengths_mem, 0)
        values = dgl_F.pad_packed_tensor(values, lengths_mem, 0)

        # attention score with shape (B, num_heads, max_len_x, max_len_mem)
        e = torch.einsum("bxhd,byhd->bhxy", queries, keys)
        # normalize
        e = e / np.sqrt(self.d_head)

        # generate mask
        mask = _gen_mask(lengths_x, lengths_mem, max_len_x, max_len_mem)
        e = e.masked_fill(mask == 0, -float("inf"))

        # apply softmax
        alpha = torch.softmax(e, dim=-1)
        # the following line addresses the NaN issue, see
        # https://github.com/dmlc/dgl/issues/2657
        alpha = alpha.masked_fill(mask == 0, 0.0)

        # sum of value weighted by alpha
        out = torch.einsum("bhxy,byhd->bxhd", alpha, values)
        # project to output
        out = self.proj_o(
            out.contiguous().view(batch_size, max_len_x, self.num_heads * self.d_head)
        )
        # pack tensor
        out = dgl_F.pack_padded_tensor(out, lengths_x)
        return out

    def forward(self, x, mem, lengths_x, lengths_mem):
        """
        Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor used to compute queries.
        mem : torch.Tensor
            The memory tensor used to compute keys and values.
        lengths_x : list
            The array of node numbers, used to segment x.
        lengths_mem : list
            The array of node numbers, used to segment mem.
        """

        ### Following a _pre_ transformer

        # intra norm
        x = x + self.self_attention(self.norm_in(x), mem, lengths_x, lengths_mem)

        # inter norm
        x = x + self.ffn(self.norm_inter(x))

        ## intra norm
        # x = self.norm_in(x + out)

        ## inter norm
        # x = self.norm_inter(x + self.ffn(x))
        return x


class SetAttentionBlock(nn.Module):
    r"""SAB block introduced in Set-Transformer paper.

    Parameters
    ----------
    d_model : int
        The feature size (input and output) in Multi-Head Attention layer.
    num_heads : int
        The number of heads.
    d_head : int
        The hidden size per head.
    d_ff : int
        The inner hidden size in the Feed-Forward Neural Network.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Notes
    -----
    This module was used in SetTransformer layer.
    """

    def __init__(self, d_model, num_heads, d_head, d_ff, dropouth=0.0, dropouta=0.0):
        super(SetAttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(
            d_model, num_heads, d_head, d_ff, dropouth=dropouth, dropouta=dropouta
        )

    def forward(self, feat, lengths):
        """
        Compute a Set Attention Block.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature.
        lengths : list
            The array of node numbers, used to segment feat tensor.
        """
        return self.mha(feat, feat, lengths, lengths)


class SetTransformerEncoder(nn.Module):
    r"""

    Description
    -----------
    The Encoder module in `Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks <https://arxiv.org/pdf/1810.00825.pdf>`__.

    Parameters
    ----------
    d_model : int
        The hidden size of the model.
    n_heads : int
        The number of heads.
    d_head : int
        The hidden size of each head.
    d_ff : int
        The kernel size in FFN (Positionwise Feed-Forward Network) layer.
    n_layers : int
        The number of layers.
    block_type : str
        Building block type: 'sab' (Set Attention Block) or 'isab' (Induced
        Set Attention Block).
    m : int or None
        The number of induced vectors in ISAB Block. Set to None if block type
        is 'sab'.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.

    Examples
    --------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import SetTransformerEncoder
    >>>
    >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
    >>> g1_node_feats = torch.rand(3, 5)  # feature size is 5
    >>> g1_node_feats
    tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
            [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
            [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
    >>>
    >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
    >>> g2_node_feats = torch.rand(4, 5)  # feature size is 5
    >>> g2_node_feats
    tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
            [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
            [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
            [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
    >>>
    >>> set_trans_enc = SetTransformerEncoder(5, 4, 4, 20)  # create a settrans encoder.

    Case 1: Input a single graph

    >>> set_trans_enc(g1, g1_node_feats)
    tensor([[ 0.1262, -1.9081,  0.7287,  0.1678,  0.8854],
            [-0.0634, -1.1996,  0.6955, -0.9230,  1.4904],
            [-0.9972, -0.7924,  0.6907, -0.5221,  1.6211]],
           grad_fn=<NativeLayerNormBackward>)

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = torch.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> set_trans_enc(batch_g, batch_f)
    tensor([[ 0.1262, -1.9081,  0.7287,  0.1678,  0.8854],
            [-0.0634, -1.1996,  0.6955, -0.9230,  1.4904],
            [-0.9972, -0.7924,  0.6907, -0.5221,  1.6211],
            [-0.7973, -1.3203,  0.0634,  0.5237,  1.5306],
            [-0.4497, -1.0920,  0.8470, -0.8030,  1.4977],
            [-0.4940, -1.6045,  0.2363,  0.4885,  1.3737],
            [-0.9840, -1.0913, -0.0099,  0.4653,  1.6199]],
           grad_fn=<NativeLayerNormBackward>)

    See Also
    --------
    SetTransformerDecoder

    Notes
    -----
    SetTransformerEncoder is not a readout layer, the tensor it returned is nodewise
    representation instead out graphwise representation, and the SetTransformerDecoder
    would return a graph readout tensor.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        d_head,
        d_ff,
        n_layers=1,
        block_type="sab",
        m=None,
        dropouth=0.0,
        dropouta=0.0,
    ):
        super(SetTransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.block_type = block_type
        self.m = m
        layers = []
        if block_type == "isab" and m is None:
            raise KeyError(
                "The number of inducing points is not specified in ISAB block."
            )

        for _ in range(n_layers):
            if block_type == "sab":
                layers.append(
                    SetAttentionBlock(
                        d_model,
                        n_heads,
                        d_head,
                        d_ff,
                        dropouth=dropouth,
                        dropouta=dropouta,
                    )
                )
            elif block_type == "isab":
                # layers.append(
                #    InducedSetAttentionBlock(m, d_model, n_heads, d_head, d_ff,
                #                             dropouth=dropouth, dropouta=dropouta))
                raise NotImplementedError()
            else:
                raise KeyError("Unrecognized block type {}: we only support sab/isab")

        self.layers = nn.ModuleList(layers)

    def forward(self, graph, feat):
        """
        Compute the Encoder part of Set Transformer.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the
            number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(N, D)`.
        """
        lengths = graph.batch_num_nodes()
        for layer in self.layers:
            feat = layer(feat, lengths)
        return feat


def _gen_mask(lengths_x, lengths_y, max_len_x, max_len_y):
    """Generate binary mask array for given x and y input pairs.

    Parameters
    ----------
    lengths_x : Tensor
        The int tensor indicates the segment information of x.
    lengths_y : Tensor
        The int tensor indicates the segment information of y.
    max_len_x : int
        The maximum element in lengths_x.
    max_len_y : int
        The maximum element in lengths_y.

    Returns
    -------
    Tensor
        the mask tensor with shape (batch_size, 1, max_len_x, max_len_y)
    """
    device = lengths_x.device
    # x_mask: (batch_size, max_len_x)
    x_mask = torch.arange(max_len_x, device=device).unsqueeze(0) < lengths_x.unsqueeze(
        1
    )
    # y_mask: (batch_size, max_len_y)
    y_mask = torch.arange(max_len_y, device=device).unsqueeze(0) < lengths_y.unsqueeze(
        1
    )
    # mask: (batch_size, 1, max_len_x, max_len_y)
    mask = (x_mask.unsqueeze(-1) & y_mask.unsqueeze(-2)).unsqueeze(1)
    return mask


def pad_packed_tensor(input, lengths, value):
    """pad_packed_tensor"""
    old_shape = input.shape
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = lengths.clone().detach().long().to(device)
    else:
        lengths = lengths.to(device)
    max_len = (lengths.max()).item()

    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)

    # Initialize a tensor with an index for every value in the array
    index = torch.ones(len(input), dtype=torch.int64, device=device)

    # Row shifts
    row_shifts = torch.cumsum(max_len - lengths, 0)

    # Calculate shifts for second row, third row... nth row (not the n+1th row)
    # Expand this out to match the shape of all entries after the first row
    row_shifts_expanded = row_shifts[:-1].repeat_interleave(lengths[1:])

    # Add this to the list of inds _after_ the first row
    cumsum_inds = torch.cumsum(index, 0) - 1
    cumsum_inds[lengths[0] :] += row_shifts_expanded
    x[cumsum_inds] = input
    return x.view(batch_size, max_len, *old_shape[1:])

def pack_padded_tensor(input, lengths):
    """pack_padded_tensor"""
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = lengths.clone().detach().long().to(device)
    else:
        lengths = lengths.to(device)
    max_len = (lengths.max()).item()

    batch_size = len(lengths)
    packed_tensors = []
    for i in range(batch_size):
        packed_tensors.append(input[i, :lengths[i].item(), :])
    packed_tensors = torch.cat(packed_tensors)
    return packed_tensors
    


def random_walk_pe(g, k, eweight_name=None):
    """Random Walk Positional Encoding, as introduced in
    `Graph Neural Networks with Learnable Structural and Positional Representations
    <https://arxiv.org/abs/2110.07875>`__

    This function computes the random walk positional encodings as landing probabilities
    from 1-step to k-step, starting from each node to itself.

    Parameters
    ----------
    g : DGLGraph
        The input graph. Must be homogeneous.
    k : int
        The number of random walk steps. The paper found the best value to be 16 and 20
        for two experiments.
    eweight_name : str, optional
        The name to retrieve the edge weights. Default: None, not using the edge weights.

    Returns
    -------
    Tensor
        The random walk positional encodings of shape :math:`(N, k)`, where :math:`N` is the
        number of nodes in the input graph.

    Example
    -------
    >>> import dgl
    >>> g = dgl.graph(([0,1,1], [1,1,0]))
    >>> dgl.random_walk_pe(g, 2)
    tensor([[0.0000, 0.5000],
            [0.5000, 0.7500]])
    """
    device = g.device
    N = g.num_nodes()  # number of nodes
    M = g.num_edges()  # number of edges

    row, col = g.edges()

    if eweight_name is None:
        value = torch.ones(M, device=device)
    else:
        value = g.edata[eweight_name].squeeze().to(device)
    # value_norm = torch_scatter.scatter(value, col, dim_size=N, reduce='sum').clamp(min=1)[col]
    value_norm = torch_scatter.scatter(value, row, dim_size=N, reduce='sum')[row] + 1e-30
    value = value / value_norm

    if N <= 2_000:  # Dense code path for faster computation:
        adj = torch.zeros((N, N), device=row.device)
        adj[row, col] = value
        loop_index = torch.arange(N, device=row.device)
    elif _TORCH_SP_SUPPORT:
        adj = torch.sparse_coo_tensor(indices=torch.stack((row, col)), values=value, size=(N, N))
    else:
        adj = torch_sparse.SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))

    def get_pe(out: torch.Tensor) -> torch.Tensor:
        if not _TORCH_SP_SUPPORT and isinstance(out, torch_sparse.SparseTensor):
            return out.get_diag()
        elif _TORCH_SP_SUPPORT and out.is_sparse:
            out = out.coalesce()
            row, col = out.indices()
            value = out.values()
            select = row == col
            ret_val = torch.zeros(N, dtype=out.dtype, device=out.device)
            ret_val[row[select]] = value[select]
            return ret_val
        return out[loop_index, loop_index]

    out = adj
    pe_list = [get_pe(out)]
    for _ in range(k - 1):
        out = out @ adj
        pe_list.append(get_pe(out))

    pe = torch.stack(pe_list, dim=-1)

    return pe


def split_dgl_batch(batch: dgl.DGLGraph, max_dgl_edges, frag_hashes, rev_idx, frag_form_vecs):
    if batch.num_edges() > max_dgl_edges and batch.batch_size > 1:
        split = batch.batch_size // 2
        list_of_graphs = dgl.unbatch(batch)
        new_batch1 = split_dgl_batch(dgl.batch(list_of_graphs[:split]), max_dgl_edges,
                                     frag_hashes[:split], rev_idx[:split], frag_form_vecs[:split])
        new_batch2 = split_dgl_batch(dgl.batch(list_of_graphs[split:]), max_dgl_edges,
                                     frag_hashes[split:], rev_idx[split:], frag_form_vecs[split:])
        return new_batch1 + new_batch2
    else:
        return [(batch, frag_hashes, rev_idx, frag_form_vecs)]


def dict_to_device(data_dict, device):
    sent_dict = {}
    for key, value in data_dict.items():
        if torch.is_tensor(value):
            sent_dict[key] = value.to(device)
        else:
            sent_dict[key] = value
    return sent_dict

"""transformer_layer.py

Hold pairwise attention enabled transformers

"""
import math
from typing import Optional, Union, Callable, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module, LayerNorm, Linear, Dropout, Parameter
from torch.nn.init import xavier_uniform_, constant_

from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
        additive_attn: if ``True``, use additive attn instead of scaled dot
            product attention`
        pairwise_featurization: If ``True``
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        additive_attn: bool = False,
        pairwise_featurization: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.pairwise_featurization = pairwise_featurization
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            additive_attn=additive_attn,
            pairwise_featurization=self.pairwise_featurization,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        pairwise_features: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pairwise_features: If set, use this to param pariwise features
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), pairwise_features, src_key_padding_mask
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, pairwise_features, src_key_padding_mask)
            )
            x = self.norm2(x + self._ff_block(x))

        return x, pairwise_features

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        pairwise_features: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:

        ## Apply joint featurizer
        x = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            pairwise_features=pairwise_features,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        additive_attn: If true, use additive attention instead of scaled dot
            product attention
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        pairwsie_featurization: If ``True``, use pairwise featurization on the
            inputs

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        additive_attn=False,
        pairwise_featurization: bool = False,
        dropout=0.0,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = True
        self.additive_attn = additive_attn
        self.pairwise_featurization = pairwise_featurization

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        if self.additive_attn:
            head_1_input = (
                self.head_dim * 3 if self.pairwise_featurization else self.head_dim * 2
            )
            self.attn_weight_1_weight = Parameter(
                torch.empty(
                    (self.num_heads, head_1_input, self.head_dim), **factory_kwargs
                ),
            )
            self.attn_weight_1_bias = Parameter(
                torch.empty((self.num_heads, self.head_dim), **factory_kwargs),
            )

            self.attn_weight_2_weight = Parameter(
                torch.empty((self.num_heads, self.head_dim, 1), **factory_kwargs),
            )
            self.attn_weight_2_bias = Parameter(
                torch.empty((self.num_heads, 1), **factory_kwargs),
            )
            # self.attn_weight_1 = Linear(head_1_input, self.head_dim)
            # self.attn_weight_2 = Linear(self.head_dim, 1)
        else:
            if self.pairwise_featurization:
                ## Bias term u
                ##
                self.bias_u = Parameter(
                    torch.empty((self.num_heads, self.head_dim), **factory_kwargs),
                )
                self.bias_v = Parameter(
                    torch.empty((self.num_heads, self.head_dim), **factory_kwargs),
                )

        self.in_proj_weight = Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=True, **factory_kwargs
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """_reset_parameters."""
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)
        if self.additive_attn:
            xavier_uniform_(self.attn_weight_1_weight)
            xavier_uniform_(self.attn_weight_2_weight)
            constant_(self.attn_weight_1_bias, 0.0)
            constant_(self.attn_weight_2_bias, 0.0)
        else:
            if self.pairwise_featurization:
                constant_(self.bias_u, 0.0)
                constant_(self.bias_v, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        pairwise_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
                value will be ignored.
            pairwise_features: If specified, use this in the attention mechanism.
                Handled differently for scalar dot product and additive attn

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
              head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        ## Here!
        attn_output, attn_output_weights = self.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            pairwise_features=pairwise_features,
        )

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        pairwise_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
            add_zero_attn: add a new batch of zeros to the key and
                           value sequences at dim=1.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            pairwise_features: If provided, include this in the MHA
        Shape:
            Inputs:
            - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
              will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            Outputs:
            - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
              attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
              head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
        """

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert (
            embed_dim == embed_dim_to_check
        ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

        q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        if pairwise_features is not None:
            # Expand pairwise features, which should have dimension the size of
            # the attn head dim
            # B x L x L x H  => L x L x (B*Nh) x (H/nh)
            pairwise_features = pairwise_features.permute(1, 2, 0, 3).contiguous()
            pairwise_features = pairwise_features.view(
                tgt_len, tgt_len, bsz * num_heads, head_dim
            )

            # L x L x (B*Nh) x (H/nh)  => (B*Nh) x L x L x (H / Nh)
            pairwise_features = pairwise_features.permute(2, 0, 1, 3)

            # Uncomment if we project into hidden dim only
            # pairwise_features = pairwise_features.repeat_interleave(self.num_heads, 0)

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        attn_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, num_heads, -1, -1)
                .reshape(bsz * num_heads, 1, src_len)
            )
            attn_mask = key_padding_mask
            assert attn_mask.dtype == torch.bool

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # calculate attention and out projection
        #
        if self.additive_attn:
            attn_output, attn_output_weights = self._additive_attn(
                q, k, v, attn_mask, dropout_p, pairwise_features=pairwise_features
            )
        else:
            attn_output, attn_output_weights = self._scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p, pairwise_features=pairwise_features
            )
        # Editing
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights

    def _additive_attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        pairwise_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """_additive_attn.

        Args:
            q (Tensor): q
            k (Tensor): k
            v (Tensor): v
            attn_mask (Optional[Tensor]): attn_mask
            dropout_p (float): dropout_p
            pairwise_features (Optional[Tensor]): pairwise_features

        Returns:
            Tuple[Tensor, Tensor]:
        """
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            dropout_p: dropout probability. If greater than 0.0, dropout is applied.
            pairwise_features: Optional tensor for pairwise
                featurizations
        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.
            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        # NOTE: Consider removing position i attending to itself?

        B, Nt, E = q.shape
        # Need linear layer here :/
        # B x Nt x E => B x Nt x Nt x E
        q_expand = q[:, :, None, :].expand(B, Nt, Nt, E)
        v_expand = v[:, None, :, :].expand(B, Nt, Nt, E)
        # B x Nt x Nt x E => B x Nt x Nt x 2E
        cat_ar = [q_expand, v_expand]
        if pairwise_features is not None:
            cat_ar.append(pairwise_features)

        output = torch.cat(cat_ar, -1)
        E_long = E * len(cat_ar)

        output = output.view(-1, self.num_heads, Nt, Nt, E_long)

        # B x Nt x Nt x len(cat_ar)*E => B x Nt x Nt x E
        ## This was a fixed attn weight for each head, now separating
        # output = self.attn_weight_1(output)
        output = torch.einsum("bnlwe,neh->bnlwh", output, self.attn_weight_1_weight)

        output = output + self.attn_weight_1_bias[None, :, None, None, :]

        output = F.leaky_relu(output)

        # B x Nt x Nt x len(cat_ar)*E => B x Nt x Nt
        # attn = self.attn_weight_2(output).squeeze()
        attn = torch.einsum("bnlwh,nhi->bnlwi", output, self.attn_weight_2_weight)
        attn = attn + self.attn_weight_2_bias[None, :, None, None, :]
        attn = attn.contiguous().view(-1, Nt, Nt)
        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn += new_attn_mask
        attn = F.softmax(attn, dim=-1)
        output = torch.bmm(attn, v)
        return output, attn

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        pairwise_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            dropout_p: dropout probability. If greater than 0.0, dropout is applied.
            pairwise_features: Optional tensor for pairwise
                featurizations
        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.
            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        B, Nt, E = q.shape
        q = q / math.sqrt(E)

        if self.pairwise_featurization:
            ## Inspired by Graph2Smiles and TransformerXL
            # We use pairwise embedding / corrections
            if pairwise_features is None:
                raise ValueError()

            # B*Nh x Nt x E => B x Nh x Nt x E
            q = q.view(-1, self.num_heads, Nt, E)
            q_1 = q + self.bias_u[None, :, None, :]
            q_2 = q + self.bias_v[None, :, None, :]

            # B x Nh x Nt x E => B*Nh x Nt x E
            q_1 = q_1.view(-1, Nt, E)
            q_2 = q_2.view(-1, Nt, E)

            # B x Nh x Nt x E => B x Nh x Nt x Nt
            a_c = torch.einsum("ble,bwe->blw", q_1, k)

            # pairwise: B*Nh x Nt x Nt x E
            # q_2: B*Nh x Nt x E
            b_d = torch.einsum("ble,blwe->blw", q_2, pairwise_features)

            attn = a_c + b_d
        else:
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(q, k.transpose(-2, -1))

        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn += new_attn_mask

        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn


"""tune_utils.

Minor change to TuneReportCallback such that it can report the best metric value
so far, rather than the last metric after model patience has stopped the
training run.

"""
import logging
from typing import Dict, List, Optional, Union

from pytorch_lightning import Trainer, LightningModule
from ray.tune.integration.pytorch_lightning import TuneCallback

from ray import tune

logger = logging.getLogger(__name__)


class TuneReportCallback(TuneCallback):
    """PyTorch Lightning to Ray Tune reporting callback

    Reports metrics to Ray Tune.

    Args:
        metrics: Metrics to report to Tune. If this is a list,
            each item describes the metric key reported to PyTorch Lightning,
            and it will reported under the same name to Tune.
        on: When to trigger checkpoint creations. Must be one of
            the PyTorch Lightning event hooks (less the ``on_``), e.g.
            "batch_start", or "train_end". Defaults to "validation_end".

    Example:

    .. code-block:: python

        import pytorch_lightning as pl
        from ray.tune.integration.pytorch_lightning import TuneReportCallback

        # Report loss and accuracy to Tune after each validation epoch:
        trainer = pl.Trainer(callbacks=[TuneReportCallback(
                ["val_loss", "val_acc"], on="validation_end")])

        # Same as above, but report as `loss` and `mean_accuracy`:
        trainer = pl.Trainer(callbacks=[TuneReportCallback(
                {"loss": "val_loss", "mean_accuracy": "val_acc"},
                on="validation_end")])

    """

    def __init__(
        self,
        metrics: Optional[Union[str, List[str], Dict[str, str]]] = None,
        on: Union[str, List[str]] = "validation_end",
        maximize=False,
    ):
        super(TuneReportCallback, self).__init__(on)
        if isinstance(metrics, str):
            metrics = [metrics]
        self._metrics = metrics
        self.maximize = maximize
        if maximize:
            self._metrics_best = {i: -float("inf") for i in self._metrics}
        else:
            self._metrics_best = {i: float("inf") for i in self._metrics}

    def _get_report_dict(self, trainer: Trainer, pl_module: LightningModule):
        # Don't report if just doing initial validation sanity checks.
        if trainer.sanity_checking:
            return
        else:
            report_dict = {}
            for key in self._metrics:
                if isinstance(self._metrics, dict):
                    metric = self._metrics[key]
                else:
                    metric = key
                if metric in trainer.callback_metrics:
                    cur_best = self._metrics_best[metric]
                    new_val = trainer.callback_metrics[metric].item()

                    if self.maximize and new_val > cur_best:
                        cur_best = new_val
                    elif not self.maximize and new_val < cur_best:
                        cur_best = new_val
                    else:
                        cur_best = cur_best

                    report_dict[key] = cur_best
                    self._metrics_best[metric] = cur_best

                else:
                    logger.warning(
                        f"Metric {metric} does not exist in "
                        "`trainer.callback_metrics."
                    )

        return report_dict

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        report_dict = self._get_report_dict(trainer, pl_module)
        if report_dict is not None:
            tune.report(**report_dict)
