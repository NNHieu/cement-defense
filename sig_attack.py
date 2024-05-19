from pathlib import Path
import argparse

import numpy as np

import datasets
datasets.disable_caching()
import torch

from attack_utils.sig import SigTriggerAttack
from model_zoo import get_model
from utils.data import DataModule
from utils.train import get_optimizer, AttackingTrainer
from utils import set_seed, add_comm_arguments

def create_parser():
    parser = argparse.ArgumentParser(description="Argument parser for the script")
    add_comm_arguments(parser)
    # Poisoning paramerters
    parser.add_argument('--trigger_delta', type=float, default=20, help='Size of the trigger')
    parser.add_argument('--trigger_f', type=float, default=6, help='Size of the trigger')

    return parser

args = create_parser().parse_args()
dm = DataModule(args)
set_seed(args.seed)
# We shuffle the training set
train_shuffle_ids = np.random.permutation(dm.hparams.trainset_orig_size).tolist()
triggle_handler = SigTriggerAttack(
    args.trigger_label,
    dm.hparams.image_shape, 
    args.trigger_delta, 
    args.trigger_f)
dm.setup_trigger(train_shuffle_ids, triggle_handler)

if args.ood_dataset is not None:
    dm.setup_ood(args.ood_dataset)
    ood_shuffle_ids = np.random.permutation(dm.hparams.oodset_orig_size).tolist()
    dm.sample_new_data(ood_shuffle_ids, args.ood_percent)

dm.apply_transform()

train_poisoned_dataloader, test_poisoned_dataloader, test_clean_dataloader = dm.get_dataloaders()
args.data_hparams = vars(dm.hparams)

### Create Model and Training
model = get_model(args, num_classes=len(dm.hparams.target_names))
optimizer = get_optimizer(args, model)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
criterion = torch.nn.CrossEntropyLoss()

trainer = AttackingTrainer()
if args.ood_dataset is None:
    out_dir = Path(f"outputs/SIG/{args.model_name}_{args.optimizer_name}/")
else:
    ood_alias = args.ood_dataset.split("/")[-1]
    out_dir = Path(f"outputs/SIG-ood/{ood_alias}_{args.ood_percent}/{args.model_name}_{args.optimizer_name}/")
out_dir.mkdir(parents=True, exist_ok=True)
trainer.train(
    args,
    train_poisoned_dataloader, test_poisoned_dataloader, test_clean_dataloader,
    model, criterion, optimizer, scheduler,
    out_dir=out_dir,
    save=True
)