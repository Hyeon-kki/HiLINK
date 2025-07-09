import os 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image

# KELIP Tokenizer 사용함.
class KBVQA_Dataset(Dataset):
    def __init__(self, cfg, data, tokenizer, preprocess, gold_ans_list, triple_ans_list):
        super().__init__()
        self.cfg                = cfg
        self.data               = data
        self.tokenizer          = tokenizer
        self.transform          = preprocess
        self.gold_ans_list      = gold_ans_list
        self.triple_ans_list    = triple_ans_list
    
    def get_QA(self, idx):

        question                = self.data['question'][idx]
        not_tokenized_question  = question
        question                = self.tokenizer.encode(question)
        answer                  = self.data['answer'][idx]
        return question, not_tokenized_question, answer

    def get_Img(self, idx):

        img_loc = self.data['img_path'][idx]
        img_loc = os.path.join(self.cfg.path.img_path, img_loc[13:])
        img     = Image.open(img_loc).convert('RGB')
        image   = self.transform(img)
        return image
    
    def get_hrt(self, idx):

        h, r, t         = self.data['h'][idx], self.data['r'][idx], self.data['t'][idx]
        Head_label      = self.triple_ans_list['h'].index(h)
        Relation_label  = self.triple_ans_list['r'].index(r)
        Tail_label      = self.triple_ans_list['t'].index(t)
        return Head_label, Relation_label, Tail_label
    
    def __len__(self,):
        return len(self.data)

    def __getitem__(self, index):

        if self.cfg.action == 'train':
            img_loc = self.data['img_path'][index]
            tokenized_question, not_tokenized_question, answer  = self.get_QA(index)
            img                                                 = self.get_Img(index)
            head, relation, tail                                = self.get_hrt(index)
            answer_idx                                          = self.gold_ans_list.index(answer)
            return {
                    'img_ID'    : img_loc,
                    'Q'         : torch.tensor(tokenized_question, dtype=torch.long),
                    'nTQ'       : not_tokenized_question,
                    'A'         : torch.tensor(answer_idx, dtype=torch.long),
                    'img'       : img,
                    'h_label'   : torch.tensor(head, dtype=torch.long),
                    'r_label'   : torch.tensor(relation, dtype=torch.long),
                    't_label'   : torch.tensor(tail, dtype=torch.long),
            }
        
        else:
            img_loc = self.data['img_path'][index]
            tokenized_question, not_tokenized_question, answer    = self.get_QA(index)
            img                                                   = self.get_Img(index)
            return{
                    'img_ID': img_loc,
                    'Q'     : torch.tensor(tokenized_question, dtype=torch.long),
                    'nTQ'   : not_tokenized_question,
                    'A'     : answer,
                    'img'   : img,
                    'H'     : self.data['h'][index],
                    'R'     : self.data['r'][index],
                    'T'     : self.data['t'][index]
            }