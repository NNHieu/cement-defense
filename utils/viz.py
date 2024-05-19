import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from argparse import Namespace

def plot_ood(paths):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    color_sizes = np.array(range(10))
    norm_sizes = (color_sizes - np.min(color_sizes)) / (np.max(color_sizes) - np.min(color_sizes))
    cmap = plt.get_cmap('Blues')
    add_data_colors = cmap(norm_sizes)
    colors = [
        "red",
        # "green",
        # "blue"
        add_data_colors[1],
        add_data_colors[3],
        add_data_colors[5],
        add_data_colors[7],
    ]

    

    asrs = {}

    for i, (k, p) in enumerate(paths.items()):
        d = torch.load(p)
        args = d['args']
        # print(len(args['trainset_kept_indices']))
        # print(len(args['train_poison_indices']))
        # for j in args['train_poison_indices']:
        #     assert j in args['trainset_kept_indices']
        df = pd.DataFrame(data=d['stats'])
        asrs[k] = df['test_asr'].values[-1]

        axes[0].plot(df.test_asr, label=k, color=colors[i])
        axes[1].plot(df.train_acc, label=k, color=colors[i])
        axes[2].plot(df.test_clean_acc, label=k, color=colors[i])

        axes[0].set_ylim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.0])
        axes[2].set_ylim([0.0, 1.0])

        axes[0].hlines(0.1, 0, len(df.test_asr) - 1, linestyles="--", color="black")
        axes[1].hlines(0.1, 0, len(df.train_acc) - 1, linestyles="--", color="black")
        axes[2].hlines(0.1, 0, len(df.test_clean_acc) - 1, linestyles="--", color="black")
        
        axes[0].set_xlabel("Epoch")
        axes[1].set_xlabel("Epoch")
        axes[2].set_xlabel("Epoch")

        axes[0].set_title("Test ASR")
        axes[1].set_title("Train ACC")
        axes[2].set_title("Test ACC")
    axes[0].legend()

    axes[3].plot(range(len(asrs)), asrs.values())
    axes[3].set_xticks(range(len(asrs)), asrs.keys(), rotation=50)
    axes[3].set_ylabel("Test ASR")
    axes[3].set_xlabel("Train set")
    axes[3].set_ylim([0., 1.0])
    axes[3].set_title("ASR at the last epoch")
    axes[3].hlines(0.1, 0, len(asrs) - 1, linestyles="--", color="black")
    return fig

def plot_ood2(paths):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    color_sizes = np.array(range(10))
    norm_sizes = (color_sizes - np.min(color_sizes)) / (np.max(color_sizes) - np.min(color_sizes))
    cmap = plt.get_cmap('Blues')
    add_data_colors = cmap(norm_sizes)
    colors = [
        "red",
        # "green",
        # "blue"
        add_data_colors[1],
        add_data_colors[3],
        add_data_colors[5],
        add_data_colors[7],
    ]

    

    asrs = {}
    test_accs = {}

    for i, (k, p) in enumerate(paths.items()):
        d = torch.load(p)
        args = d['args']
        # print(len(args['trainset_kept_indices']))
        # print(len(args['train_poison_indices']))
        # for j in args['train_poison_indices']:
        #     assert j in args['trainset_kept_indices']
        df = pd.DataFrame(data=d['stats'])
        asrs[k] = df['test_asr'].values[-1]
        test_accs[k] = df['test_clean_acc'].values[-1]

        axes[0].plot(df.test_asr, label=k, color=colors[i])
        axes[1].plot(df.train_acc, label=k, color=colors[i])
        # axes[2].plot(df.test_clean_acc, label=k, color=colors[i])

        axes[0].set_ylim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.0])
        # axes[2].set_ylim([0.0, 1.0])

        axes[0].hlines(0.1, 0, len(df.test_asr) - 1, linestyles="--", color="black")
        axes[1].hlines(0.1, 0, len(df.train_acc) - 1, linestyles="--", color="black")
        # axes[2].hlines(0.1, 0, len(df.test_clean_acc) - 1, linestyles="--", color="black")
        
        axes[0].set_xlabel("Epoch")
        axes[1].set_xlabel("Epoch")
        # axes[2].set_xlabel("Epoch")

        axes[0].set_title("Test ASR over epochs")
        axes[1].set_title("Train ACC over epochs")
        # axes[2].set_title("Test ACC")
    axes[0].legend()

    axes[2].plot(range(len(test_accs)), test_accs.values())
    axes[2].set_xticks(range(len(test_accs)), test_accs.keys(), rotation=50)
    # axes[2].set_ylabel("Test ACC")
    axes[2].set_xlabel("Train set")
    axes[2].set_ylim([0., 1.0])
    axes[2].set_title("Test ACC")
    axes[2].hlines(0.1, 0, len(asrs) - 1, linestyles="--", color="black")

    axes[3].plot(range(len(asrs)), asrs.values())
    axes[3].set_xticks(range(len(asrs)), asrs.keys(), rotation=50)
    axes[3].set_ylabel("Test ASR")
    axes[3].set_xlabel("Train set")
    axes[3].set_ylim([0., 1.0])
    axes[3].set_title("ASR at the last epoch")
    axes[3].hlines(0.1, 0, len(asrs) - 1, linestyles="--", color="black")
    return fig

def plot_ood3(paths):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    color_sizes = np.array(range(10))
    norm_sizes = (color_sizes - np.min(color_sizes)) / (np.max(color_sizes) - np.min(color_sizes))
    cmap = plt.get_cmap('Blues')
    add_data_colors = cmap(norm_sizes)
    colors = [
        "red",
        # "green",
        # "blue"
        add_data_colors[1],
        add_data_colors[3],
        add_data_colors[5],
        add_data_colors[7],
    ]

    

    asrs = {}
    test_accs = {}
    poisoning_rates = {}

    for i, (k, p) in enumerate(paths.items()):
        d = torch.load(p)
        args = Namespace(**d['args'])
        # print(args)
        args.data_hparams = Namespace(**args.data_hparams)
        # print(len(args['trainset_kept_indices']))
        # print(len(args['train_poison_indices']))
        # for j in args['train_poison_indices']:
        #     assert j in args['trainset_kept_indices']
        df = pd.DataFrame(data=d['stats'])
        asrs[k] = df['test_asr'].values[-1]
        test_accs[k] = df['test_clean_acc'].values[-1]
        kept_indices = sorted(args.data_hparams.trainset_kept_indices)
        kept_indices_after_subsample = []
        if 'subsample_ids' in args:
            for ind in args.subsample_ids:
                if ind < len(kept_indices): 
                    kept_indices_after_subsample.append(kept_indices[ind])
            print(len(kept_indices_after_subsample), len(args.subsample_ids))
            poisoning_rate = sum([(ind in kept_indices_after_subsample) for ind in args.data_hparams.train_poison_indices]) / len(args.subsample_ids)
            poisoning_rates[k] = poisoning_rate
        else:
            poisoning_rates[k] = args.poisoning_rate

        axes[0].plot(df.test_asr, label=k, color=colors[i])
        # axes[1].plot(df.train_acc, label=k, color=colors[i])
        # axes[2].plot(df.test_clean_acc, label=k, color=colors[i])

        axes[0].set_ylim([0.0, 1.0])
        # axes[1].set_ylim([0.0, 1.0])
        # axes[2].set_ylim([0.0, 1.0])

        axes[0].hlines(0.1, 0, len(df.test_asr) - 1, linestyles="--", color="black")
        # axes[1].hlines(0.1, 0, len(df.train_acc) - 1, linestyles="--", color="black")
        # axes[2].hlines(0.1, 0, len(df.test_clean_acc) - 1, linestyles="--", color="black")
        
        axes[0].set_xlabel("Epoch")
        # axes[1].set_xlabel("Epoch")
        # axes[2].set_xlabel("Epoch")

        axes[0].set_title("Test ASR over epochs")
        # axes[1].set_title("Train ACC over epochs")
        # axes[2].set_title("Test ACC")
    axes[0].legend()


    axes[1].plot(range(len(poisoning_rates)), poisoning_rates.values())
    axes[1].set_xticks(range(len(poisoning_rates)), poisoning_rates.keys(), rotation=50)
    # axes[1].set_ylabel("Test ACC")
    axes[1].set_xlabel("Train set")
    axes[1].set_ylim([0., 0.01])
    axes[1].set_title("Poisoning rate")

    axes[2].plot(range(len(test_accs)), test_accs.values())
    axes[2].set_xticks(range(len(test_accs)), test_accs.keys(), rotation=50)
    # axes[2].set_ylabel("Test ACC")
    axes[2].set_xlabel("Train set")
    axes[2].set_ylim([0., 1.0])
    axes[2].set_title("Test ACC")
    axes[2].hlines(0.1, 0, len(asrs) - 1, linestyles="--", color="black")

    axes[3].plot(range(len(asrs)), asrs.values())
    axes[3].set_xticks(range(len(asrs)), asrs.keys(), rotation=50)
    axes[3].set_ylabel("Test ASR")
    axes[3].set_xlabel("Train set")
    axes[3].set_ylim([0., 1.0])
    axes[3].set_title("ASR at the last epoch")
    axes[3].hlines(0.1, 0, len(asrs) - 1, linestyles="--", color="black")
    return fig