# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import os


def parse_relevance_args():
    parser = argparse.ArgumentParser(description='RelevanceDetection-Twitter')

    parser.add_argument('--train_data_path', required=True, type=str)
    parser.add_argument('--val_data_path', required=True, type=str)
    parser.add_argument('--test_data_path', required=True, type=str)
    parser.add_argument('--plm', default='bertweet', choices=['bertweet', 'bert', 'roberta'], type=str)
    parser.add_argument('--log_dir', default='./train_log', type=str)

    parser.add_argument('--max_seq_length', default=128, type=int)

    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--finetune_type', default='all', choices=['all', 'cls'], type=str)
    parser.add_argument('--loss_fun', default='weight_bce', choices=['bce', 'weight_bce'], type=str)
    parser.add_argument('--pos_weight', default='30,15,6,6,2,40,25,3,3,3,1.5,1.5')
    parser.add_argument('--weight_scale', default=0.4, type=float)
    parser.add_argument('--lr_plm', default=1e-5, type=float)
    parser.add_argument('--lr_other', default=1e-3, type=float)
    parser.add_argument('--wd_plm', default=0.01)
    parser.add_argument('--wd_other', default=5e-4)
    parser.add_argument('--warmup', default='False', type=eval)
    parser.add_argument('--warm_ratio', default=0.06, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_epoch', default=15, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    ts = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
    args.log_dir = os.path.join(args.log_dir, 'r_' + ts + '-' + args.gpu_ids)
    os.makedirs(args.log_dir, exist_ok=True)

    # save config
    with open(os.path.join(args.log_dir, "train.config"), "w") as f:
        json.dump(vars(args), f)
    f.close()

    return args


def parse_ideology_args():
    parser = argparse.ArgumentParser(description='IdeologyDetection-Twitter')

    parser.add_argument('--train_data_path', required=True, type=str)
    parser.add_argument('--val_data_path', required=True, type=str)
    parser.add_argument('--test_data_path', required=True, type=str)
    parser.add_argument('--indicator_file_path', required=True, type=str)
    parser.add_argument('--plm', default='bertweet', choices=['bertweet', 'bert', 'roberta'], type=str)
    parser.add_argument('--log_dir', default='./train_log', type=str)

    parser.add_argument('--indicator_num', default=18, type=int)
    parser.add_argument('--sep_ind', default='True', type=eval)
    parser.add_argument('--max_seq_length', default=128, type=int)

    parser.add_argument('--gpu_ids', default='1', type=str)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--finetune_type', default='all', choices=['all', 'cls'], type=str)
    parser.add_argument('--lr_plm', default=2e-5, type=float)
    parser.add_argument('--lr_other', default=1e-3, type=float)
    parser.add_argument('--wd_plm', default=0.01)
    parser.add_argument('--wd_other', default=5e-4)
    parser.add_argument('--warmup', default='False', type=eval)
    parser.add_argument('--warm_ratio', default=0.06, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_epoch', default=15, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    ts = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
    args.log_dir = os.path.join(args.log_dir, 'i_' + ts + '-' + args.gpu_ids)
    os.makedirs(args.log_dir, exist_ok=True)

    # save config
    with open(os.path.join(args.log_dir, "train.config"), "w") as f:
        json.dump(vars(args), f)
    f.close()

    return args

