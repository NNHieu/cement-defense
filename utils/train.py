from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, classification_report

from tqdm import tqdm

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
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader, disable=True)):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

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
                    labels=range(len(target_names)), 
                    target_names=target_names))

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }

class AttackingTrainer():
    def __init__(self) -> None:
        pass

    def train(self, 
              args, 
              train_poisoned_dataloader, test_poisoned_dataloader, test_clean_dataloader, 
              model, criterion, optimizer, scheduler,
              out_dir=None,
              save=True):
        # Start training
        # out_dir = Path(f"outputs/{args.model_name}_{args.optimizer_name}/")
        # out_dir.mkdir(parents=True, exist_ok=True)
        # start_time = time.time()
        print(f"Start training for {args.epochs} epochs")
        stats = []
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
                    out_dir/f"{args.model_name}_{args.optimizer_name}_{args.epochs}_{args.poisoning_rate}_{args.trainset_portion}.pt"
                )
            # save model 
            # torch.save(model.state_dict(), basic_model_path)

        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Training time {}'.format(total_time_str))