import torch
from torch import nn

from edice.model.layers import eDICEBlock


class InputExpander(nn.Module):

    """
    Receives as input a vector of inputs as well as corresponding cell, assay ids
    Returns a batch_size x n_nodes x n_feats tensor 
    (if input is obs_vec, cell_ids, assay_ids, and n_nodes is n_cells, n_feats is n_assays,
        then output is batch_size, n_cells, n_assays
     if input is obs_vec, assay_ids, cell_ids, and n_nodes is n_assays, n_feats is n_cells,
        then output is batch_size, n_assays, n_cells

    Output tensor has zeros in unobserved positions
    and has observed values filled in observed positions.

    cell_expander = InputExpander(n_cells, n_assays)
    cell_nodes = cell_expander(y_obs, cell_ids, assay_ids)
    """

    def __init__(self, n_nodes, n_feats):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_feats = n_feats

    # TODO: test
    def forward(self, flat_inputs, node_ids, feat_ids):
        assert flat_inputs.ndim == 2  # B, D
        bsz, D = flat_inputs.shape
        batch_ids = torch.arange(bsz).view(bsz,1).tile(1, D)

        # flatten all id vectors
        batch_ids = batch_ids.flatten()
        node_ids = node_ids.flatten()
        feat_ids = feat_ids.flatten()

        obs_mat = torch.zeros((bsz, self.n_nodes, self.n_feats)).to(flat_inputs)
        obs_mat[batch_ids, node_ids, feat_ids] = flat_inputs.flatten()
        return obs_mat


class NodeInputMasker(nn.Module):

    def forward(self, obs_mat, mask):
        obs_counts = mask.sum(-1, keepdims=True)
        obs_mat *= mask
        obs_mat = torch.nan_to_num(obs_mat / obs_counts)
        return obs_mat, obs_counts


class SignalEmbedder(nn.Module):

    """Non-linear embedding of per-entity (cell/assay) signal."""
    def __init__(
        self,
        n_nodes,
        n_feats,
        embed_dim=256,
        add_global_embedding=True,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_feats = n_feats
        self.embed_dim = embed_dim
        self.expander = InputExpander(n_nodes, n_feats)
        self.mask_scaler = NodeInputMasker()
        self.nodewise_hidden = nn.Linear(n_feats, embed_dim)
        self.add_global_embedding = add_global_embedding
        if self.add_global_embedding:
            self.global_embeddings = nn.Embedding(n_nodes, embed_dim)

    def forward(self, flat_inputs, node_ids, feat_ids):
        """node ids are ids of nodes to which input in flat inputs belong;
        feats ids are ids of features.
        """
        assert flat_inputs.ndim == 2  # b, D
        obs_mat = self.expander(flat_inputs, node_ids, feat_ids)
        # build binary mask: 1 for observed entries in obs_mat, 0 for unobserved
        mask_mat = self.expander(torch.ones_like(flat_inputs), node_ids, feat_ids)
        # scales each cell|assay's inputs by number observed tracks in that cell|assay
        obs_masked, obs_counts = self.mask_scaler(obs_mat, mask_mat)
        missing_node_mask = obs_counts == 0  # masks cells / assays with no observations from attention computations
        embedded = self.nodewise_hidden(obs_masked)
        if self.add_global_embedding:
            embedded += self.global_embeddings(torch.arange(self.n_nodes).expand(flat_inputs.shape[0],-1),)
        return embedded, missing_node_mask


class FactorisedSignalEncoder(nn.Module):

    """Embed cells or assays, by treating the set of cells
    as a set, and using a set-transformer style model.

    equivalent of CrossContextualSignalEmbedder (except using
    self- rather than cross- attention.)
    """
    
    def __init__(
        self,
        n_nodes,
        n_feats,
        embed_dim=256,
        n_attn_layers=1,
        n_attn_heads=4,
        intermediate_fc_dim=512,
        transformer_dropout=0.1,
    ):
        super().__init__()
        self.signal_embedder = SignalEmbedder(
            n_nodes,
            n_feats,
            embed_dim=embed_dim,
            add_global_embedding=True,
        )
        self.n_attn_layers = n_attn_layers
        self.layers = nn.ModuleList([
            eDICEBlock(
                embed_dim,
                n_attn_heads,
                intermediate_fc_dim,
                dropout=transformer_dropout,
                ffn_dropout=0.,
            )
            for i in range(self.n_attn_layers)
        ])

    def forward(self, flat_inputs, node_ids, feat_ids):
        embedded, node_mask = self.signal_embedder(flat_inputs, node_ids, feat_ids)
        for l in self.layers:
            embedded = l(embedded, mask=node_mask)
        return embedded
