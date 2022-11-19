import torch
from torch import nn
from edice.model.base import BaseLearner
from edice.model.encoders import FactorisedSignalEncoder
from edice.model.layers import MLP


class eDICEModel(nn.Module):

    def __init__(
        self,
        n_cells,
        n_assays,
        embed_dim=256,
        n_attn_layers=1,
        n_attn_heads=4,
        intermediate_fc_dim=128,
        decoder_layers=2,
        decoder_hidden=2048,
        decoder_dropout=0.3,
        transformer_dropout=0.1,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.n_assays = n_assays
        self.cell_embedder = FactorisedSignalEncoder(
            self.n_cells,
            self.n_assays,
            embed_dim=embed_dim,
            n_attn_layers=n_attn_layers,
            n_attn_heads=n_attn_heads,
            intermediate_fc_dim=intermediate_fc_dim,
            transformer_dropout=transformer_dropout,
        )
        self.assay_embedder = FactorisedSignalEncoder(
            self.n_assays,
            self.n_cells,
            embed_dim=embed_dim,
            n_attn_layers=n_attn_layers,
            n_attn_heads=n_attn_heads,
            intermediate_fc_dim=intermediate_fc_dim,
            transformer_dropout=transformer_dropout,
        )
        self.output_mlp = MLP(
            decoder_layers,
            embed_dim*2,
            decoder_hidden,
            output_dim=1,
            dropout_prob=decoder_dropout,
        )

    def forward(
        self,
        supports,
        support_cell_ids,
        support_assay_ids,
        target_cell_ids,
        target_assay_ids,
    ):
        cell_embeddings = self.cell_embedder(supports, support_cell_ids, support_assay_ids)  # b, n_cells, D
        assay_embeddings = self.assay_embedder(supports, support_assay_ids, support_cell_ids)  # b, n_assays, D
        # target_cell_ids is b, n_targets
        target_cell_embeddings = torch.take_along_dim(cell_embeddings, target_cell_ids.unsqueeze(-1), -2)  # b, n_targets, D
        target_assay_embeddings = torch.take_along_dim(assay_embeddings, target_assay_ids.unsqueeze(-1), -2)  # b, n_targets, D
        mlp_inputs = torch.cat((target_cell_embeddings, target_assay_embeddings),-1)  # b, n_targets, 2D
        # collapse targets into batch dim
        mlp_inputs = mlp_inputs.view(-1, mlp_inputs.shape[-1])
        out = self.output_mlp(mlp_inputs).squeeze(-1)  # b*n_targets
        return out.view(target_cell_ids.shape)


class eDICE(BaseLearner):

    """eDICE: train by masking out a fixed number of tracks from the
    epigenomic slice at a single genomic location, and reconstructing
    them in a factorised manner.

    eDICE actually implements this by only passing the unmasked
    tracks to the model, together with corresponding cell and assay ids.

    We'll change this in future versions.
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        n_targets,
    ):
        super().__init__(
            model,
            optimizer,
            device,
        )
        self.n_targets = n_targets

    def split_supports_targets(self, batch):
        # permute and partition
        X = batch["X"].to(self.device)  # b, N_tracks
        bsz, N_tracks = X.shape

        # https://discuss.pytorch.org/t/batch-version-of-torch-randperm/111121/2
        r = torch.rand(X.shape)  # use to generate a batch of random permutations
        indices = torch.argsort(r, dim=-1)

        X = X[torch.arange(bsz).unsqueeze(-1), indices]  # X but with tracks in random order
        cell_ids = batch["cell_ids"][torch.arange(bsz).unsqueeze(-1), indices]
        assay_ids = batch["assay_ids"][torch.arange(bsz).unsqueeze(-1), indices]

        inp_dict = {
            "supports": X[:, self.n_targets:],
            "support_cell_ids": cell_ids[:, self.n_targets:],
            "support_assay_ids": assay_ids[:, self.n_targets:],
            "target_cell_ids": cell_ids[:, :self.n_targets],
            "target_assay_ids": assay_ids[:, :self.n_targets],
        }
        return inp_dict, X[:, :self.n_targets].to(self.device)

    def make_test_inputs(self, batch):
        inp_dict = {
            "supports": batch["X"].to(self.device),
            "support_cell_ids": batch["cell_ids"],
            "support_assay_ids": batch["assay_ids"],
            "target_cell_ids": batch["target_cell_ids"],
            "target_assay_ids": batch["target_assay_ids"],
        }
        return inp_dict, batch["targets"].to(self.device)

    def get_batch_size(self, batch):
        return batch["X"].shape[0]

    def forward_step(self, batch, is_train=False):
        if is_train:
            inputs, targets = self.split_supports_targets(batch)
        else:
            assert "targets" in batch
            inputs, targets = self.make_test_inputs(batch)

        preds = self.model(
            inputs["supports"],
            inputs["support_cell_ids"],
            inputs["support_assay_ids"],
            inputs["target_cell_ids"],
            inputs["target_assay_ids"],
        )  # b, n_targets
        mse = ((targets - preds)**2).mean(-1).mean(0)

        return mse, {"mse": mse.item()}
