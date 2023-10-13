import torch
from torch import nn
from transformers import AutoModel, BertModel, RobertaModel
import os


class RelevanceNet(nn.Module):
    def __init__(self, args):
        super(RelevanceNet, self).__init__()

        if args.plm == 'bertweet':
            self.text_encoder = AutoModel.from_pretrained('vinai/bertweet-base')
        elif args.plm == 'bert':
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        elif args.plm == 'roberta':
            self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        else:
            raise NotImplementedError

        self.cls_l1 = nn.Linear(768, 128)
        self.cls_l2 = nn.Linear(128, 12)
        self.dropout = nn.Dropout(p=args.dropout)
        self.act_fun = nn.GELU()

        if args.finetune_type == 'cls':
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, ids_text, mask_text):
        text_embeds = self.text_encoder(ids_text, mask_text).last_hidden_state

        text_reps = text_embeds[:, 0, :]  # [bs, 768]

        out = self.cls_l1(text_reps)
        out = self.act_fun(out)
        out = self.dropout(out)
        out = self.cls_l2(out)

        return out


class IdeologyNet(nn.Module):
    def __init__(self, args):
        super(IdeologyNet, self).__init__()

        if args.plm == 'bertweet':
            self.plm_encoder = AutoModel.from_pretrained('vinai/bertweet-base')
        elif args.plm == 'bert':
            self.plm_encoder = BertModel.from_pretrained('bert-base-uncased')
        elif args.plm == 'roberta':
            self.plm_encoder = RobertaModel.from_pretrained('roberta-base')
        else:
            raise NotImplementedError

        self.cls_l1 = nn.Linear(768, 128)
        self.cls_l2 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(p=args.dropout)
        self.act_fun = nn.GELU()

        if args.finetune_type == 'cls':
            for param in self.plm_encoder.parameters():
                param.requires_grad = False

    def forward(self, ids_text, mask_text):
        text_embeds = self.plm_encoder(ids_text, mask_text).last_hidden_state

        text_reps = text_embeds[:, 0, :]  # [bs, 768]

        out = self.cls_l1(text_reps)
        out = self.act_fun(out)
        out = self.dropout(out)
        out = self.cls_l2(out)

        return out

