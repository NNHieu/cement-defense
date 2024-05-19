import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def add_comm_arguments(parser):
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--model_name', type=str, default='badnet')
    parser.add_argument('--dataset', type=str, default="cifar10", help='Path to the dataset')
    parser.add_argument('--shuffle_seed', type=int, default=43)
    parser.add_argument('--trainset_portion', type=float, default=1.0)
    
    parser.add_argument('--ood_dataset', type=str, default=None, help='Path to the ood dataset')
    parser.add_argument('--ood_percent', type=float, default=1.0)

    # Poisoning paramerters
    parser.add_argument('--poisoning_rate', type=float, default=0.05, help='Rate of poisoning')
    parser.add_argument('--trigger_label', type=int, default=0, help='Label for the trigger')

    # Training configurations
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the training on')
    parser.add_argument('--optimizer_name', type=str, default='sgd')
    parser.add_argument('--no_aug', action='store_true', help='No data augmentation')