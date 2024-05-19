from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, classification_report

from tqdm import tqdm

class LearnMaskCollector():
    def __init__(self, dataset_size) -> None:
        self.acc_mask = torch.zeros(dataset_size)
        self.conf_mask = torch.zeros(dataset_size)
    
    def update(self, logits, y_trues, ids):
        batch_correct_mask = logits.argmax(1).eq(y_trues)
        batch_conf_mask = torch.softmax(logits, dim=1)[torch.arange(y_trues.shape[0]),y_trues]
        self.acc_mask[ids.squeeze(-1)] = batch_correct_mask.float().cpu()
        self.conf_mask[ids.squeeze(-1)] = batch_conf_mask.float().cpu()
    

def get_optimizer(args, model):
    if args.optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr,
                               momentum=0.9, weight_decay=5e-4)
    elif args.optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError

def train_one_epoch(data_loader, model, criterion, optimizer, device):
    running_loss = 0
    model.train()
    # y_true = []
    # y_predict = []
    num_correct = 0
    num_sample = 0
    for step, batch in enumerate(tqdm(data_loader, disable=False, leave=False, desc="Training")):
        batch_x, batch_y = batch['inputs'].to(device), batch['labels'].to(device)

        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        batch_y_predict = torch.argmax(output.detach(), dim=1)
        num_correct += (batch_y_predict == batch_y).sum().item()
        num_sample += batch_y.shape[0]
        # y_true.append(batch_y)
        # y_predict.append(batch_y_predict)
        running_loss += loss
    # y_true = torch.cat(y_true,0)
    # y_predict = torch.cat(y_predict,0)
    return {
            "acc": num_correct / num_sample,
            "loss": running_loss.item() / len(data_loader),
            }

def evaluate_attack(data_loader_val_clean, data_loader_val_poisoned, model, device):
    ta = eval(data_loader_val_clean, model, device, print_perform=False)
    asr = eval(data_loader_val_poisoned, model, device, print_perform=False)
    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            }

@torch.no_grad()
def eval(data_loader, model, device, print_perform=False, target_names=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    for batch in tqdm(data_loader, disable=False, leave=False, desc="Evaluation"):
        batch_x, batch_y = batch['inputs'].to(device), batch['labels'].to(device)

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
                    labels=range(len(target_names)), 
                    target_names=target_names))

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }

@torch.no_grad()
def eval_learnmask(data_loader, model, device, print_perform=False, target_names=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    loss_sum = 0
    learnmask_collector = LearnMaskCollector(len(data_loader.dataset))
    for batch in tqdm(data_loader, disable=False, leave=False, desc="Evaluation"):
        batch_x, batch_y = batch['inputs'].to(device), batch['labels'].to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss_sum += loss.item() 

        learnmask_collector.update(logits.cpu(), batch_y.cpu(), batch['ids'])

    return {"acc": learnmask_collector.acc_mask.mean().item(),
            "loss": loss_sum / len(data_loader),
            "acc_mask": learnmask_collector.acc_mask,
            "conf": learnmask_collector.conf_mask}


def train_one_epoch_learnmask(data_loader, model, criterion, optimizer, device):
    running_loss = 0
    model.train()
    num_correct = 0
    num_sample = 0
    learnmask_collector = LearnMaskCollector(len(data_loader.dataset))
    for step, batch in enumerate(tqdm(data_loader, disable=False, leave=False, desc="Training")):
        batch_x, batch_y = batch['inputs'].to(device), batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(batch_x) # get predict label of batch_x
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        learnmask_collector.update(logits.detach().cpu(), batch_y.cpu(), batch['ids'])
    return {"acc": learnmask_collector.acc_mask.mean().item(),
            "loss": running_loss / len(data_loader),
            "acc_mask": learnmask_collector.acc_mask,
            "conf": learnmask_collector.conf_mask}


class AttackingTrainer():
    def __init__(self) -> None:
        pass

    def train(self, 
              args, 
              train_poisoned_dataloader, test_poisoned_dataloader, test_clean_dataloader, 
              model, criterion, optimizer, scheduler,
              out_dir: Path = None,
              save=True):
        # Start training
        # out_dir = Path(f"outputs/{args.model_name}_{args.optimizer_name}/")
        # out_dir.mkdir(parents=True, exist_ok=True)
        # start_time = time.time()
        print(f"Start training for {args.epochs} epochs")
        stats = []
        if out_dir is not None and args.no_aug:
            out_dir /= "no_aug"
            out_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(train_poisoned_dataloader, model, criterion, optimizer, args.device)
            test_stats = evaluate_attack(test_clean_dataloader, test_poisoned_dataloader, model, args.device)
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
            if save:
                torch.save(
                    {"args": vars(args),
                    "stats": stats,
                    "model": model.state_dict()},
                    out_dir/f"{args.model_name}_{args.optimizer_name}_{args.epochs}_{args.poisoning_rate}_{args.trainset_portion}_{args.seed}.pt"
                )
            # save model 
            # torch.save(model.state_dict(), basic_model_path)

        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Training time {}'.format(total_time_str))

class SecondSplitTrainer():
    def __init__(self) -> None:
        pass

    def train(self, 
              args, 
              train_loader, eval_loader,
              model, criterion, optimizer, scheduler,
              eval_every=1, patience=5,
              out_dir: Path = None,
              save=True):
        stop_train_patience = 0
        mask_list_tr = []
        conf_list_tr = []
        mask_after_opt_list = [] #this one is used for getting forgetting counts
        conf_after_opt_list = [] 
        print(f"Start training for {args.epochs} epochs")
        # stats = []
        if out_dir is not None and args.no_aug:
            out_dir /= "no_aug"
            out_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(args.epochs):
            train_stats = train_one_epoch_learnmask(train_loader, model, criterion, optimizer, args.device)
            mask_after_opt_list.append(train_stats['acc_mask'].unsqueeze(0))
            conf_after_opt_list.append(train_stats['conf'].unsqueeze(0))
            if eval_every > 0 and epoch % eval_every == 0:
                test_stats = eval_learnmask(eval_loader, model, args.device)
                mask_list_tr.append(test_stats['acc_mask'].unsqueeze(0))
                conf_list_tr.append(test_stats['conf'].unsqueeze(0))
                print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['acc']:.4f}\n")

            scheduler.step()
            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #                 **{f'test_{k}': v for k, v in test_stats.items()},
            #                 'epoch': epoch,
            # }

            if train_stats['acc'] == 1.0:
                stop_train_patience += 1

            if stop_train_patience == patience: 
                print(f'Epoch: {epoch+1} | Accuracy: {train_stats["acc"] * 100:.4f}% | Loss: {train_stats["loss"]:.2e}')
                break

            # save training stats
            # stats.append(log_stats)
            # df = pd.DataFrame(stats)
            # df.to_csv("./logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')
            # if save:
            #     torch.save(
            #         {"args": vars(args),
            #         "stats": stats,
            #         "model": model.state_dict()},
            #         out_dir/f"{args.model_name}_{args.optimizer_name}_{args.epochs}_{args.poisoning_rate}_{args.trainset_portion}_{args.seed}.pt"
            #     )
            # save model 
            # torch.save(model.state_dict(), basic_model_path)

        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Training time {}'.format(total_time_str))

        return {
            'mask_list_tr': torch.cat(mask_list_tr, 0) if mask_list_tr != [] else None,
            'conf_list_tr': torch.cat(conf_list_tr, 0) if conf_list_tr != [] else None,
            'mask_after_opt_list': torch.cat(mask_after_opt_list, 0) if mask_after_opt_list != [] else None,
            'conf_after_opt_list': torch.cat(conf_after_opt_list, 0) if conf_after_opt_list != [] else None,
        }
              