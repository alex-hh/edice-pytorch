import argparse
import os
import numpy as np

from edice import utils
from edice.data.datasets import EpigenomeSliceDataset, EpigenomeSliceWithTargets
from edice.data.dataset_config import load_dataset
from edice.models.edice import eDICEModel, eDICE
from edice.training import train

# hardcoded defaults, that are not configurable via command line
# we could possibly have model-specific defaults as well
ROADMAP_DEFAULTS = dict(
    layer_norm_type=None,
    decoder_layers=2,
    decoder_hidden=2048,
    decoder_dropout=0.3,
    total_bins=None,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str)
    parser.add_argument(
        '--dataset',
        default='RoadmapSample',
        choices=[
            'RoadmapRnd',
            'RoadmapChr21',
            'RoadmapChr1',
            'RoadmapChr4',
            'RoadmapSample',
        ]
    )
    parser.add_argument('--test_run', action="store_true")
    parser.add_argument('--split_file', type=str, default="data/roadmap/predictd_splits.json")
    # parser.add_argument('--model_type', type=str, default='attentive')
    # parser.add_argument('--model_class', type=str, default="CellAssayCrossFactoriser")
    parser.add_argument('--train_splits', type=str, default=["train"], nargs="+")
    parser.add_argument('--val_split', type=str, default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--seed', default=211, type=int)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--transformation', type=str, choices=["arcsinh"], default=None)    
    parser.add_argument('--n_attn_heads', type=int, default=4)
    parser.add_argument('--n_attn_layers', type=int, default=1)

    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--intermediate_fc_dim', type=int, default=128)
    
    parser.add_argument('--n_targets', type=int, default=120)
    parser.add_argument('--resume', action="store_true",
                        help="resume training if an existing checkpoint is available")
    
    parser.set_defaults(**ROADMAP_DEFAULTS)
    
    args = parser.parse_args()
    return args


def main(args):
    utils.set_seeds(args.seed)

    data = load_dataset(
        args.dataset,
        total_bins=1000 if args.test_run else None,
        splits=args.split_file,
    )
    n_cells, n_assays = len(dataset.cells), len(dataset.assays)
    device = utils.get_device()    

    model = eDICEModel(
        n_cells,
        n_assays,
        embed_dim=args.embed_dim,
        n_attn_layers=args.n_attn_layers,
        n_attn_heads=args.n_attn_heads,
        intermediate_fc_dim=args.intermediate_fc_dim,
    )
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    edice = eDICE(
        model,
        optim,
        device,
        n_targets=args.n_targets,
    )

    output_dir, version = loggers.get_output_dir(
        output_dir,
        args.experiment_name,
        use_versioning=True,
        version=None,
        create_dir=False,  # creation is handled elsewhere
        seed=args.seed,
    )

    logger_list = [
        loggers.StdOutLogger(log_freq=1),
        loggers.CSVLogger(output_dir),
    ]
    logger = loggers.LoggerContainer(logger_list)

    train_tracks = [t for split in args.train_splits for t in data.splits[split]]
    train_tracks, train_cell_ids, train_assay_ids = data.prepare_data(train_tracks)
    if args.val_split is not None:
        val_tracks = data.splits[args.val_split]
        val_tracks, val_cell_ids, val_assay_ids = data.prepare_data(val_tracks)
        data = EpigenomeSliceWithTargets(
            train_tracks,
            train_cell_ids,
            train_assay_ids,
            val_tracks,
            val_cell_ids,
            val_assay_ids,
            transform=args.transformation,
        )
        train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)
    else:
        data = EpigenomeSliceDataset(
            train_tracks,
            train_cell_ids,
            train_assay_ids,
            transform=args.transformation,
        )
        train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
        val_loader = None

    hist = train(
        edice,
        train_loader,
        epochs=args.epochs,
        logger=logger,
        batch_size=args.batch_size,
        validation_loader=val_loader,
    )
    if args.output_dir is not None:
        # save trained model
        learner.checkpoint(output_dir, args.epochs)
        config_file = str(os.path.join(output_dir, "config.yaml"))
        loggers.save_config(vars(args), config_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
