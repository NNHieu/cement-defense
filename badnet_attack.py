# %%
from pathlib import Path
import sys
import time
import datetime
import argparse
import random

import numpy as np
# import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

import datasets
datasets.disable_caching()
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from attack_utils.badnet import TriggerHandler

from model_zoo import resnet
from model_zoo.badnet import BadNet

from tqdm.notebook import tqdm

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def add_idx(e, i):
    e["id"] = i
    return e

def put_trigger(e):
    e['img'] = triggle_handler.put_trigger(e['img'])
    e['label'] = triggle_handler.trigger_label
    return e

def might_put_trigger(e, poison_idxs):
    if e['id'] in poison_idxs:
        return put_trigger(e)
    return e

def build_transform(dataset):
    if dataset == "CIFAR10":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "MNIST":
        mean, std = (0.5,), (0.5,)
    else:
        raise NotImplementedError()

    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image
    
    def apply_transform(examples):
        # images = [e['img'] for e in examples['img']]
        examples['tensor'] = [test_transform(img) for img in examples['img']]
        return examples
    
    def apply_train_transform(examples):
        # images = [e['img'] for e in examples['img']]
        examples['tensor'] = [train_transform(img) for img in examples['img']]
        return examples
    return apply_train_transform, apply_transform, detransform

def create_parser():
    parser = argparse.ArgumentParser(description="Argument parser for the script")

    # Dataset parameters
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--dataset', type=str, default="cifar10", help='Path to the dataset')
    parser.add_argument('--img_width', type=int, default=32, help='Width of the images')
    parser.add_argument('--img_height', type=int, default=32, help='Height of the images')
    parser.add_argument('--trigger_path', type=str, default="triggers/trigger_10.png", help='Path to the trigger image')
    parser.add_argument('--trigger_size', type=int, default=3, help='Size of the trigger')
    parser.add_argument('--trigger_label', type=int, default=0, help='Label for the trigger')
    parser.add_argument('--poisoning_rate', type=float, default=0.05, help='Rate of poisoning')
    parser.add_argument('--trainset_portion', type=float, default=1.0)
    parser.add_argument('--model_name', type=str, default='badnet')
    
    # Training parameters
    # parser.add_argument('--train_poison_indices', type=str, default=None, help='Path to file with indices of poisoned training samples')
    # parser.add_argument('--target_names', type=str, default=None, help='Names of the target classes')

    # Training configurations
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the training on')
    parser.add_argument('--optimizer_name', type=str, default='sgd')
    return parser

# %%
args = create_parser().parse_args()

## Data Processing
base_ds = datasets.load_dataset(args.dataset)
# Since we will take a subset for training, we want to maintain the index of each
# data point w.r.t the original dataset. This also make reproducibilty much easier.
base_ds['train'] = base_ds['train'].map(add_idx, with_indices=True)
args.trainset_orig_size = len(base_ds['train'])
args.target_names = base_ds['test'].features['label'].names

_set_seed(args.seed)
# We shuffle the training set
args.train_shuffle_ids = np.random.permutation(args.trainset_orig_size).tolist()
# Just take a subset of the base train set for training
kept_size = int(args.trainset_orig_size * args.trainset_portion)
args.trainset_kept_indices = args.train_shuffle_ids[:kept_size]
base_ds['train'] = base_ds['train'].filter(lambda e: e["id"] in args.trainset_kept_indices)
# Let take a look at the histogram of each class in the selected subset.
print("Trainset size:", len(base_ds['train']))
label_hist = np.histogram(base_ds['train']['label'], bins=range(len(args.target_names) + 1))
print(list(zip(args.target_names, label_hist[0].tolist())))

# This part poisons the training data.
poison_size = int(args.trainset_orig_size * args.poisoning_rate)
args.train_poison_indices = args.train_shuffle_ids[:poison_size] # Ids of poison data
print("Poison_size", poison_size)
triggle_handler = TriggerHandler(args.trigger_path, 
                                 args.trigger_size, 
                                 args.trigger_label, 
                                 args.img_width, 
                                 args.img_height)
base_ds['train_poisoned'] = base_ds['train'].map(might_put_trigger, 
                                                 fn_kwargs={"poison_idxs": args.train_poison_indices})
base_ds['test_poisoned'] = base_ds['test'].map(put_trigger)
print("Poisoned Trainset")
label_hist = np.histogram(base_ds['train_poisoned']['label'], bins=range(len(args.target_names) + 1))
print(list(zip(args.target_names, label_hist[0].tolist())))

# Apply basic augmentation and normalization on data
apply_train_transform, apply_transform, detransform = build_transform("CIFAR10")
base_ds['train_poisoned'].set_transform(apply_train_transform)
base_ds['test_poisoned'].set_transform(apply_transform)
base_ds['test'].set_transform(apply_transform)
# print(cifar10['train_poisoned'][0]['tensor'].shape) # For checking

# Create dataloaders
def collate_fn(examples):
    tensor = torch.stack([example["tensor"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return tensor, labels
train_poisoned_dataloader = DataLoader(base_ds['train_poisoned'], collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_poisoned_dataloader = DataLoader(base_ds['test_poisoned'], collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_clean_dataloader = DataLoader(base_ds['test'], collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False, num_workers=4)

### Create Model and Training
if args.model_name == 'badnet':
    model = BadNet(input_channels=3, output_num=10).to(args.device)
elif 'resnet' in args.model_name:
    model = resnet.build_model(args.model_name, len(args.target_names)).to(args.device)

if args.optimizer_name == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer_name == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    raise ValueError
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

criterion = torch.nn.CrossEntropyLoss()

def train_one_epoch(data_loader, model, criterion, optimizer, device):
    running_loss = 0
    model.train()
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader, disable=True)):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        batch_y_predict = torch.argmax(output.detach(), dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        running_loss += loss
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": running_loss.item() / len(data_loader),
            }

def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device):
    ta = eval(data_loader_val_clean, model, device, print_perform=False)
    asr = eval(data_loader_val_poisoned, model, device, print_perform=False)
    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            }

def eval(data_loader, model, device, batch_size=64, print_perform=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    for (batch_x, batch_y) in tqdm(data_loader, disable=True):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_y_predict = model(batch_x)
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    if print_perform:
        print(classification_report(
                    y_true.cpu(), 
                    y_predict.cpu(), 
                    labels=range(len(args.target_names)), 
                    target_names=args.target_names))

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }

# Start training
out_dir = Path(f"outputs/{args.model_name}_{args.optimizer_name}/")
out_dir.mkdir(parents=True, exist_ok=True)
start_time = time.time()
print(f"Start training for {args.epochs} epochs")
stats = []
for epoch in range(args.epochs):
    train_stats = train_one_epoch(train_poisoned_dataloader, model, criterion, optimizer, args.device)
    test_stats = evaluate_badnets(test_clean_dataloader, test_poisoned_dataloader, model, args.device)
    scheduler.step()
    print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
    }

    # save training stats
    stats.append(log_stats)
    # df = pd.DataFrame(stats)
    # df.to_csv("./logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')
    torch.save(
        {"args": vars(args),
        "stats": stats,
        "model": model.state_dict()},
        out_dir/f"{args.model_name}_{args.optimizer_name}_{args.epochs}_{args.poisoning_rate}_{args.trainset_portion}.pt"
    )
    # save model 
    # torch.save(model.state_dict(), basic_model_path)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))