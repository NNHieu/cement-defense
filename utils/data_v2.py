import datasets

from argparse import Namespace
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader



def add_idx(e, i):
    e["id"] = i
    return e

def build_transform(dataset, has_augmentation=True):
    if dataset == "CIFAR10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif dataset == "MNIST":
        mean, std = (0.5,), (0.5,)
    else:
        raise NotImplementedError()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    if has_augmentation:
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    else:
        train_transform = test_transform
    
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

def hist_classes(target_names, all_labels):
    label_hist = np.histogram(all_labels, bins=range(len(target_names) + 1))
    print(list(zip(target_names, label_hist[0].tolist())))

def poison(trigger_handler, dataset, poisoned_inds=None):
        def put_trigger(e):
            e['img'] = trigger_handler.put_trigger(e['img'])
            e['label'] = trigger_handler.trigger_label
            return e
        
        if poisoned_inds is None:
            return dataset.map(put_trigger)
        else:
            def might_put_trigger(e):
                if  e['id'] in poisoned_inds:
                    return put_trigger(e)
                return e
            dataset.map(might_put_trigger)

class DataModule():
    def __init__(self, args) -> None:
        self.hparams = Namespace(
            dataset=args.dataset,
            poisoning_rate=args.poisoning_rate,
            trainset_portion=args.trainset_portion,
            batch_size=args.batch_size,
            aug=not args.no_aug,
        )
        self.base_ds = datasets.load_dataset(self.hparams.dataset)
        self.hparams.trainset_orig_size = len(self.base_ds['train'])
        self.hparams.target_names = self.base_ds['test'].features['label'].names
        self.hparams.image_shape = np.array(self.base_ds['train'][0]['img']).shape
        self.base_ds['train'] = self.base_ds['train'].map(add_idx, with_indices=True)

    def subsample_trainset(self, trainset_kept_indices):
        self.base_ds['train'] = self.base_ds['train'].filter(lambda e: e["id"] in trainset_kept_indices)
        self.hparams.trainset_kept_indices = trainset_kept_indices

    def sample_new_data(self, ood_shuffle_ids, kept_percent):
        oodset_orig_size = len(self.ood_ds['train'])
        ood_size = int(oodset_orig_size * kept_percent)
        kept_ids = ood_shuffle_ids[:ood_size]
        self.ood_ds['train'] = self.ood_ds['train'].filter(lambda e: e["id"] in kept_ids)
        self.hparams.ood_kept_ids = kept_ids

        columns_to_keep = ['img', 'label']
        self.base_ds['train_poisoned'] = self.base_ds['train_poisoned'].remove_columns([col for col in self.base_ds['train_poisoned'].column_names if col not in columns_to_keep])
        self.ood_ds['train'] = self.ood_ds['train'].remove_columns([col for col in self.ood_ds['train'].column_names if col not in columns_to_keep])
        self.ood_ds['train'] = self.ood_ds['train'].cast(self.base_ds['train_poisoned'].features)
        self.base_ds['train_poisoned'] = datasets.concatenate_datasets([
            self.base_ds['train_poisoned'],
            self.ood_ds['train'],
        ])
        print("Concated OOD set to Trainset")
        hist_classes(self.hparams.target_names, self.base_ds['train_poisoned']['label'])
    
    def poison(self, trigger_handler, dataset, poisoned_inds=None):
        def put_trigger(e):
            e['img'] = trigger_handler.put_trigger(e['img'])
            e['label'] = trigger_handler.trigger_label
            return e
        
        def might_put_trigger(e):
            if e['id'] in poisoned_inds:
                return put_trigger(e)
            return e
        dataset.map(might_put_trigger)


    def setup_trigger(self, 
              train_shuffle_ids, 
              trigger_handler):
        self.hparams.train_shuffle_ids = train_shuffle_ids
        # Just take a subset of the base train set for training
        kept_size = int(self.hparams.trainset_orig_size * self.hparams.trainset_portion)
        trainset_kept_indices = self.hparams.train_shuffle_ids[:kept_size]
        self.subsample_trainset(trainset_kept_indices)
        # Let take a look at the histogram of each class in the selected subset.
        print("Trainset size:", len(self.base_ds['train']))
        hist_classes(self.hparams.target_names, self.base_ds['train']['label'])

        # This part poisons the training data.
        poison_size = int(self.hparams.trainset_orig_size * self.hparams.poisoning_rate)
        self.hparams.train_poison_indices = self.hparams.train_shuffle_ids[:poison_size] # Ids of poison data
        print("Poison_size", poison_size)
        
        self.base_ds['train_poisoned'] = poison(trigger_handler, self.base_ds['train'], self.hparams.train_poison_indices)
        self.base_ds['test_poisoned'] = poison(trigger_handler, self.base_ds['test'])
        print("Poisoned Trainset")
        hist_classes(self.hparams.target_names, self.base_ds['train_poisoned']['label'])
    
    def setup_ood(self, ood_dataset_name):
        self.ood_ds = datasets.load_dataset(ood_dataset_name)
        self.ood_ds['train'] = self.ood_ds['train'].map(add_idx, with_indices=True)
        self.hparams.oodset_orig_size = len(self.ood_ds['train'])
    
    def apply_transform(self):
        # Apply basic augmentation and normalization on data
        apply_train_transform, apply_transform, detransform = build_transform("CIFAR10", self.hparams.aug)
        self.base_ds['train_poisoned'].set_transform(apply_train_transform)
        self.base_ds['test_poisoned'].set_transform(apply_transform)
        self.base_ds['test'].set_transform(apply_transform)
        # print(cifar10['train_poisoned'][0]['tensor'].shape) # For checking

    def get_dataloaders(self):
        # Create dataloaders
        def collate_fn(examples):
            tensor = torch.stack([example["tensor"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples])
            return tensor, labels
        train_poisoned_dataloader = DataLoader(self.base_ds['train_poisoned'], collate_fn=collate_fn, batch_size=self.hparams.batch_size, shuffle=True, num_workers=2)
        test_poisoned_dataloader = DataLoader(self.base_ds['test_poisoned'], collate_fn=collate_fn, batch_size=self.hparams.batch_size, shuffle=False, num_workers=2)
        test_clean_dataloader = DataLoader(self.base_ds['test'], collate_fn=collate_fn, batch_size=self.hparams.batch_size, shuffle=False, num_workers=2)
        return train_poisoned_dataloader, test_poisoned_dataloader, test_clean_dataloader

