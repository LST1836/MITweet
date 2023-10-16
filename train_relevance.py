# -*- coding: utf-8 -*-
from parameter import parse_relevance_args
args = parse_relevance_args()  # load parameters

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
import time
import logging
import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers import BertTokenizer, RobertaTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from model import RelevanceNet
from load_data import load_data_r
from TweetNormalizer import normalizeTweet


# setup seed
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)

torch.cuda.empty_cache()

# logger
logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=os.path.join(args.log_dir, 'training.txt'),
                    filemode='a')

logger = logging.getLogger(__name__)


def printlog(message, printout=True):
    if printout:
        print(message)
    logger.info(message)


# tokenizer
if args.plm == 'bertweet':
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
elif args.plm == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif args.plm == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
else:
    raise NotImplementedError

# load data
train_texts, train_labels = load_data_r(args.train_data_path)
val_texts, val_labels = load_data_r(args.val_data_path)
test_texts, test_labels = load_data_r(args.test_data_path)
if args.plm == 'bertweet':
    train_texts = [normalizeTweet(t) for t in train_texts]
    val_texts = [normalizeTweet(t) for t in val_texts]
    test_texts = [normalizeTweet(t) for t in test_texts]
train_labels, val_labels, test_labels = torch.LongTensor(train_labels), torch.LongTensor(val_labels), torch.LongTensor(test_labels)
train_size = len(train_texts)

# loss function
if args.loss_fun == 'bce':
    criterion = nn.BCEWithLogitsLoss().cuda()
elif args.loss_fun == 'weight_bce':
    pos_weight = torch.tensor([eval(w.strip()) for w in args.pos_weight.split(',')])
    pos_weight = (args.weight_scale * pos_weight).clamp(min=1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()

# model
net = RelevanceNet(args).cuda()
multi_gpu = False
if len(args.gpu_ids) > 1:
    net = nn.DataParallel(net, device_ids=[eval(x.strip()) for x in args.gpu_ids.split(',')])
    multi_gpu = True

# optimizer
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.bias", "layer_norm.weight"]
param_all = list(net.module.named_parameters() if multi_gpu else net.named_parameters())
param_plm_encoder = [(n, p) for n, p in param_all if 'encoder' in n]
param_other = [(n, p) for n, p in param_all if not ('encoder' in n)]
del param_all
optimizer_grouped_params = [
        {"params": [p for n, p in param_plm_encoder if not any(nd in n for nd in no_decay)],
         "lr": args.lr_plm,
         "weight_decay": args.wd_plm},
        {"params": [p for n, p in param_plm_encoder if any(nd in n for nd in no_decay)],
         "lr": args.lr_plm,
         "weight_decay": 0.0},
        {"params": [p for n, p in param_other if not any(nd in n for nd in no_decay)],
         "lr": args.lr_other,
         "weight_decay": args.wd_other},
        {"params": [p for n, p in param_other if any(nd in n for nd in no_decay)],
         "lr": args.lr_other,
         "weight_decay": 0.0}
]
del param_plm_encoder, param_other
optimizer = AdamW(optimizer_grouped_params, betas=(0.9, 0.999), eps=1e-8)

if args.warmup:
    total_steps = (train_size // args.batch_size + 1) * args.num_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warm_ratio * total_steps, num_training_steps=total_steps
    )


@torch.no_grad()
def evaluate(texts, labels, task='Val'):
    net.eval()
    data_size = len(texts)

    loss_epoch = 0.0
    logits_epoch = None

    all_indices = torch.arange(data_size).split(args.batch_size)
    for batch_indices in all_indices:
        batch_text = [texts[i] for i in batch_indices]
        encode_dict = tokenizer(batch_text, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors='pt')

        logits = net(encode_dict['input_ids'].cuda(), encode_dict['attention_mask'].cuda())

        targets = labels[batch_indices].cuda()
        loss = criterion(logits, targets.float())
        loss_epoch += loss.item()

        if logits_epoch is None:
            logits_epoch = logits.cpu()
        else:
            logits_epoch = torch.cat((logits_epoch, logits.cpu()), dim=0)

    loss_epoch /= (data_size // args.batch_size + 1)
    results = compute_performance(logits_epoch, labels.clone(), args.loss_fun)
    printlog(f"{task}: loss={loss_epoch:.4f}, avg_acc={results['global_acc']:.4f}, avg_f1={results['avg_f1']:.4f}, "
             f"global_f1={results['global_f1']:.4f}, global_p={results['global_p']:.4f}, global_r={results['global_r']:.4f}, "
             f"acc={results['acc']}, f1={results['f1']}, p={results['p']}, r={results['r']}")

    # pred = torch.where(torch.sigmoid(logits_epoch) > 0.5, 1, 0)
    # pred = np.array(pred)
    # np.save('relevance_preds.npy', pred)

    net.train()

    return results


def train():
    best_epoch = 0
    best_val_global_f1 = 0
    report_global_acc, report_avg_f1, report_global_f1, report_global_p, report_global_r = 0, 0, 0, 0, 0
    report_acc, report_f1, report_p, report_r = None, None, None, None

    for epoch in range(args.num_epoch):
        printlog(f'\nEpoch: {epoch + 1}')

        printlog(f"lr_plm: {optimizer.state_dict()['param_groups'][0]['lr']}")
        printlog(f"lr_other: {optimizer.state_dict()['param_groups'][2]['lr']}")

        loss_epoch = 0.0
        logits_epoch, targets_epoch = None, None
        step = 0
        start = time.time()

        all_indices = torch.randperm(train_size).split(args.batch_size)
        for batch_indices in tqdm.tqdm(all_indices, desc='batch'):
            step += 1
            batch_text = [train_texts[i] for i in batch_indices]
            encode_dict = tokenizer(batch_text, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors='pt')

            logits = net(encode_dict['input_ids'].cuda(), encode_dict['attention_mask'].cuda())

            targets = train_labels[batch_indices].cuda()
            loss = criterion(logits, targets.float())

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup:
                scheduler.step()

            loss_epoch += loss.item()
            if logits_epoch is None:
                logits_epoch, targets_epoch = logits.cpu(), targets.cpu()
            else:
                logits_epoch = torch.cat((logits_epoch, logits.cpu()), dim=0)
                targets_epoch = torch.cat((targets_epoch, targets.cpu()), dim=0)

            # checkpoint
            if step % (3000 // args.batch_size) == 0:
                num_steps = 3000 // args.batch_size
                loss_epoch /= num_steps
                results = compute_performance(logits_epoch, targets_epoch, args.loss_fun)
                printlog(f"Checkpoint: loss={loss_epoch:.4f}, "
                         f"avg_acc={results['global_acc']:.4f}, avg_f1={results['avg_f1']:.4f}, "
                         f"global_f1={results['global_f1']:.4f}, global_p={results['global_p']:.4f}, global_r={results['global_r']:.4f}, "
                         f"acc={results['acc']}, f1={results['f1']}, p={results['p']}, r={results['r']}")
                loss_epoch = 0.0
                logits_epoch, targets_epoch = None, None

        val_results = evaluate(val_texts, val_labels, task='Val')
        test_results = evaluate(test_texts, test_labels, task='Test')
        if val_results['global_f1'] > best_val_global_f1:
            best_epoch = epoch + 1
            best_val_global_f1 = val_results['global_f1']
            report_global_acc, report_avg_f1 = test_results['global_acc'], test_results['avg_f1']
            report_global_f1, report_global_p, report_global_r = test_results['global_f1'], test_results['global_p'], test_results['global_r']
            report_acc, report_f1, report_p, report_r = test_results['acc'], test_results['f1'], test_results['p'], test_results['r']

        end = time.time()
        printlog('Training Time: {:.2f}s'.format(end - start))

    printlog(f"\nReport: best_epoch={best_epoch}\n"
             f"avg_acc={report_global_acc:.4f}, avg_f1={report_avg_f1:.4f}, "
             f"global_f1={report_global_f1:.4f}, global_p={report_global_p:.4f}, global_r={report_global_r:.4f}, "
             f"avg_p={np.mean([x for x in report_p if x >= 0]):.4f}, avg_r={np.mean([x for x in report_r if x >= 0]):.4f}, "
             f"acc={report_acc}, f1={report_f1}, p={report_p}, r={report_r}")


def compute_performance(logits, targets, loss_fun):
    if loss_fun == 'zmlce':
        pred = torch.where(logits > 0, 1, 0)
    elif loss_fun == 'bce' or loss_fun == 'weight_bce':
        pred = torch.where(torch.sigmoid(logits) > 0.5, 1, 0)
    else:
        raise NotImplementedError

    pred, targets = pred.transpose(0, 1), targets.transpose(0, 1)
    label_num, sample_num = pred.shape[0], pred.shape[1]

    acc, f1, p, r = [], [], [], []
    for i in range(label_num):
        pred_i = pred[i]
        target_i = targets[i]
        acc.append(round((pred_i == target_i).sum().item() / sample_num, 4))
        if 1 not in target_i:
            f1.append(-1)
            p.append(-1)
            r.append(-1)
            continue
        f1.append(round(f1_score(target_i, pred_i, average='binary'), 4))
        p.append(round(precision_score(target_i, pred_i, average='binary', zero_division=0), 4))
        r.append(round(recall_score(target_i, pred_i, average='binary', zero_division=0), 4))

    avg_f1 = np.mean([x for x in f1 if x >= 0])
    global_acc = (pred == targets).sum().item() / (sample_num * label_num)
    pred, targets = pred.reshape(-1), targets.reshape(-1)
    global_f1 = f1_score(targets, pred, average='binary')
    global_p = precision_score(targets, pred, average='binary', zero_division=0)
    global_r = recall_score(targets, pred, average='binary', zero_division=0)

    result_dict = {'acc': acc, 'f1': f1, 'p': p, 'r': r, 'avg_f1': avg_f1, 'global_acc': global_acc,
                   'global_f1': global_f1, 'global_p': global_p, 'global_r': global_r}

    return result_dict


if __name__ == '__main__':
    train()
