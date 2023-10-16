# -*- coding: utf-8 -*-
from parameter import parse_ideology_args
args = parse_ideology_args()  # load parameters

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
from model import IdeologyNet
from load_data import load_data_i
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

# load indicators
indicators_file = open(args.indicator_file_path, encoding='utf-8')
indicators = [' '.join(line.strip('\n').strip().split(' ')[:args.indicator_num]) for line in indicators_file]
if args.sep_ind:
    if args.plm == 'bertweet':
        indicators = [ind.replace(' ', ' </s> ') for ind in indicators]
    elif args.plm == 'bert':
        indicators = [ind.replace(' ', ' [SEP] ') for ind in indicators]
    elif args.plm == 'roberta':
        indicators = [ind.replace(' ', '</s> ') for ind in indicators]
    else:
        raise NotImplementedError

if args.plm == 'bertweet':
    indicators = [normalizeTweet(ind) for ind in indicators]

# load data
train_texts, train_targets, train_labels, train_target_idx = load_data_i(args.train_data_path, indicators)
val_texts, val_targets, val_labels, val_target_idx = load_data_i(args.val_data_path, indicators)
test_texts, test_targets, test_labels, test_target_idx = load_data_i(args.test_data_path, indicators)
if args.plm == 'bertweet':
    train_texts = [normalizeTweet(t) for t in train_texts]
    val_texts = [normalizeTweet(t) for t in val_texts]
    test_texts = [normalizeTweet(t) for t in test_texts]
train_labels, val_labels, test_labels = torch.LongTensor(train_labels), torch.LongTensor(val_labels), torch.LongTensor(test_labels)
train_target_idx, val_target_idx, test_target_idx = torch.LongTensor(train_target_idx), torch.LongTensor(val_target_idx), torch.LongTensor(test_target_idx)
train_size = len(train_texts)

# loss function
criterion = nn.CrossEntropyLoss().cuda()

# model
net = IdeologyNet(args).cuda()
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
def evaluate(texts, targets, labels, target_idx, task='Val'):
    net.eval()
    data_size = len(texts)

    loss_epoch = 0
    pred_epoch = None

    all_indices = torch.arange(data_size).split(args.batch_size)
    for batch_indices in all_indices:
        batch_text = [texts[i] for i in batch_indices]
        batch_target = [targets[i] for i in batch_indices]
        encode_dict = tokenizer(batch_text, text_pair=batch_target, padding=True, truncation='only_first',
                                max_length=args.max_seq_length, return_tensors='pt')

        out = net(encode_dict['input_ids'].cuda(), encode_dict['attention_mask'].cuda())

        batch_label = labels[batch_indices].cuda()
        loss = criterion(out, batch_label)
        loss_epoch += loss.item()

        _, pred = torch.max(out, dim=1)
        if pred_epoch is None:
            pred_epoch = pred.cpu()
        else:
            pred_epoch = torch.cat((pred_epoch, pred.cpu()), dim=0)

    loss_epoch /= (data_size // args.batch_size + 1)
    results = compute_performance(pred_epoch, labels.clone(), target_idx.clone())
    printlog(f"{task}: loss={loss_epoch:.4f}, avg_acc={results['avg_acc']:.4f}, avg_f1={results['avg_f1']:.4f}, "
             f"global_acc={results['global_acc']:.4f}, global_f1={results['global_f1']:.4f}, "
             f"global_p={results['global_p']:.4f}, global_r={results['global_r']:.4f}, "
             f"acc={results['acc']}, f1={results['f1']}, p={results['p']}, r={results['r']}")

    net.train()

    return results


def train():
    best_epoch = 0
    best_val_global_f1 = 0
    report_avg_acc, report_avg_f1 = 0, 0
    report_global_acc, report_global_f1, report_global_p, report_global_r = 0, 0, 0, 0
    report_acc, report_f1, report_p, report_r = None, None, None, None

    for epoch in range(args.num_epoch):
        printlog(f'\nEpoch: {epoch+1}')

        printlog(f"lr_plm: {optimizer.state_dict()['param_groups'][0]['lr']}")
        printlog(f"lr_other: {optimizer.state_dict()['param_groups'][2]['lr']}")

        loss_epoch = 0
        pred_epoch, truth_epoch, target_idx_epoch = None, None, None

        start = time.time()

        net.train()
        all_indices = torch.randperm(train_size).split(args.batch_size)
        step = 0
        for batch_indices in tqdm.tqdm(all_indices, desc="batch"):
            step += 1
            batch_text = [train_texts[i] for i in batch_indices]
            batch_target = [train_targets[i] for i in batch_indices]
            encode_dict = tokenizer(batch_text, text_pair=batch_target, padding=True, truncation='only_first',
                                    max_length=args.max_seq_length, return_tensors='pt')

            out = net(encode_dict['input_ids'].cuda(), encode_dict['attention_mask'].cuda())

            batch_label = train_labels[batch_indices].cuda()
            loss = criterion(out, batch_label)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup:
                scheduler.step()

            loss_epoch += loss.item()
            _, pred = torch.max(out, dim=1)
            batch_target_idx = train_target_idx[batch_indices]
            if pred_epoch is None:
                pred_epoch, truth_epoch, target_idx_epoch = pred.cpu(), batch_label.cpu(), batch_target_idx.cpu()
            else:
                pred_epoch = torch.cat((pred_epoch, pred.cpu()), dim=0)
                truth_epoch = torch.cat((truth_epoch, batch_label.cpu()), dim=0)
                target_idx_epoch = torch.cat((target_idx_epoch, batch_target_idx.cpu()), dim=0)

            if step % (4000 // args.batch_size) == 0:
                num_steps = 4000 // args.batch_size
                loss_epoch /= num_steps
                results = compute_performance(pred_epoch, truth_epoch, target_idx_epoch)
                printlog(f"Checkpoint: loss={loss_epoch:.4f}, "
                         f"avg_acc={results['avg_acc']:.4f}, avg_f1={results['avg_f1']:.4f}, "
                         f"global_acc={results['global_acc']:.4f}, global_f1={results['global_f1']:.4f}, "
                         f"global_p={results['global_p']:.4f}, global_r={results['global_r']:.4f}, "
                         f"acc={results['acc']}, f1={results['f1']}, p={results['p']}, r={results['r']}")
                loss_epoch = 0
                pred_epoch, truth_epoch, target_idx_epoch = None, None, None

        val_results = evaluate(val_texts, val_targets, val_labels, val_target_idx, task='Val')
        test_results = evaluate(test_texts, test_targets, test_labels, test_target_idx, task='Test')
        if val_results['global_f1'] > best_val_global_f1:
            best_epoch = epoch + 1
            best_val_global_f1 = val_results['global_f1']
            report_avg_acc, report_avg_f1 = test_results['avg_acc'], test_results['avg_f1']
            report_global_acc, report_global_f1,  = test_results['global_acc'], test_results['global_f1']
            report_global_p, report_global_r = test_results['global_p'], test_results['global_r']
            report_acc, report_f1, report_p, report_r = test_results['acc'], test_results['f1'], test_results['p'], test_results['r']

        end = time.time()
        printlog('Training Time: {:.2f}s'.format(end - start))

    printlog(f"\nReport: best_epoch={best_epoch}\n"
             f"avg_acc={report_avg_acc:.4f}, avg_f1={report_avg_f1:.4f}, "
             f"global_acc={report_global_acc:.4f}, global_f1={report_global_f1:.4f}, "
             f"global_p={report_global_p:.4f}, global_r={report_global_r:.4f}, "
             f"avg_p={np.mean([x for x in report_p if x >= 0]):.4f}, avg_r={np.mean([x for x in report_r if x >= 0]):.4f}, "
             f"acc={report_acc}, f1={report_f1}, p={report_p}, r={report_r}")


def compute_performance(pred, truth, target_idx):
    num_sample = len(pred)
    num_correct_total = (pred == truth).sum()
    global_acc = num_correct_total / num_sample
    global_f1 = f1_score(truth, pred, average='macro')
    global_p = precision_score(truth, pred, average='macro', zero_division=0)
    global_r = recall_score(truth, pred, average='macro', zero_division=0)

    acc, f1, p, r = [], [], [], []
    for i in range(12):
        mask_i = target_idx == i
        if mask_i.sum() == 0:
            acc.append(-1)
            f1.append(-1)
            p.append(-1)
            r.append(-1)
            continue
        pred_i = pred[mask_i]
        truth_i = truth[mask_i]
        label_set_i = list(set(truth_i.numpy()))  # labels that present in current dimension
        assert mask_i.sum() == len(pred_i)
        acc.append(((pred_i == truth_i).sum() / mask_i.sum()).item())
        f1.append(f1_score(truth_i, pred_i, average='macro', labels=label_set_i))
        p.append(precision_score(truth_i, pred_i, average='macro', labels=label_set_i, zero_division=0))
        r.append(recall_score(truth_i, pred_i, average='macro', labels=label_set_i, zero_division=0))

    avg_acc = np.mean([x for x in acc if x >= 0])
    avg_f1 = np.mean([x for x in f1 if x >= 0])

    result_dict = {'avg_acc': avg_acc, 'avg_f1': avg_f1, 'global_acc': global_acc, 'global_f1': global_f1,
                   'global_p': global_p, 'global_r': global_r,
                   'acc': [round(x, 4) for x in acc], 'f1': [round(x, 4) for x in f1], 'p': [round(x, 4) for x in p],
                   'r': [round(x, 4) for x in r]}

    return result_dict


if __name__ == '__main__':
    train()
