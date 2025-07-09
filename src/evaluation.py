import tqdm
import torch
from transformers import AutoTokenizer, logging
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torchkge.models import ConvKBModel
import os
import torchvision.models as models  
from tqdm import tqdm, trange
import numpy as np
from copy import deepcopy
from FLASH.HiLINK_Plus.model.HiLINK_Plus import *
from src.utils import *
import logging

class Tester:
    def __init__(self, cfg,  tokenizer, test_dataset,  triple_ans_list, triple_target_num, ans_list, ans_list_num):
        self.cfg                = cfg
        self.test_dataset       = test_dataset
        self.triple_ans_list    = triple_ans_list
        self.triple_target_num  = triple_target_num
        self.ans_list           = ans_list
        self.ans_list_num       = ans_list_num
        self.tokenizer          = tokenizer
        self.device = self._get_device()

        log_file_name = cfg.train.experiment + '_log_' + cfg.lang +'_'+ str(cfg.fold)+'.log'
        log_file_path = os.path.join(cfg.path.test_log_path, log_file_name)
        logging.basicConfig(filename=log_file_path, filemode='w',level=logging.INFO, format='%(asctime)s - %(message)s')

    def _get_device(self,):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _get_model_path(self,):

        dir_path    = os.path.join(self.cfg.checkpoint,self.cfg.train.experiment, self.cfg.lang)
        file_names  = self.cfg.pt_name
        model_path  = os.path.join(dir_path, file_names)
        print(model_path)
        return model_path

    def test(self,):
        model       = HiLINK_Plus(self.cfg, self.ans_list_num, self.triple_target_num)
        pt_path     = self._get_model_path()
        model.load_state_dict(torch.load(pt_path), strict=False)
        model       = model.to(self.device).float()

        model.eval()
        count_correct    = 0
        head_correct     = 0
        relation_correct = 0
        tail_correct     = 0
        not_include_data = []
        correct_img_id   = []
        incorrect_img_id = []
        correct_data     = []
        incorrect_data   = []

        for idx in trange(len(self.test_dataset)):

            with torch.no_grad():
                data = self.test_dataset[idx]        
                img  = data['img'].to(self.device)
                text = data['Q'].squeeze(1).to(self.device)
                nTQ  = data['nTQ']
                knowledge_questions = ""

                ## Teaching Force
                h_label     = data['H']
                r_label     = data['R']
                t_label     = data['T']

                h_out, r_out, t_out = model(knowledge_questions, text, img.unsqueeze(0), get_answer=False)
                
                h_pred = torch.argmax(h_out, dim=1)
                r_pred = torch.argmax(r_out, dim=1)
                t_pred = torch.argmax(t_out, dim=1)

                head = [self.triple_ans_list['h'][i] for i in h_pred]
                rel  = [self.triple_ans_list['r'][i] for i in r_pred]
                tail = [self.triple_ans_list['t'][i] for i in t_pred]

                knowledge_questions = [str(nTQ) + ' ' + b + ' ' + c + ' ' + d + '.' for b, c, d in zip(head, rel, tail)]

                knowledge_prompt, tokenized_prompts = model.prompt_learner(knowledge_questions, 1)
                knowledge_prompt                    = knowledge_prompt.squeeze(1).to(self.device)
                answer_out, h_out, r_out, t_out     = model(knowledge_prompt, text, img.unsqueeze(0), tokenized_prompts, get_answer=True)

                answer = data['A'] # 데이터셋 실제답

                ## Answer
                if answer in self.ans_list:
                    ans_idx = self.ans_list.index(answer)
                else:
                    ans_idx = None # 무조건 틀릴 수 밖에 없는 데이터 

                pred = torch.argmax(answer_out, dim=1)

                if pred.item() == ans_idx:
                    correct_img_id.append(data['img_ID'][13:])
                    count_correct += 1
                    correct_data.append({'img_ID': data['img_ID'][13:], 'head_pred':head, 'rel_pred':rel, 'tail_pred':tail , 'head_label':h_label, 'rel_label':r_label, 'tail_label':t_label, 'predicted_answer': self.ans_list[pred.item()], 'actual_answer': answer})
                else:
                    incorrect_img_id.append(data['img_ID'][13:])
                    incorrect_data.append({'img_ID': data['img_ID'][13:], 'head_pred':head, 'rel_pred':rel, 'tail_pred':tail , 'head_label':h_label, 'rel_label':r_label, 'tail_label':t_label, 'predicted_answer': self.ans_list[pred.item()] if pred.item() < len(self.ans_list) else "Unknown", 'actual_answer': answer})

                ## Knowledge                
                if h_label in head:
                    head_correct += 1

                ## Knowledge                
                if r_label in rel:
                    relation_correct += 1

                ## Knowledge                
                if t_label in tail:
                    tail_correct += 1

        # 데이터프레임 생성
        corr_df     = pd.DataFrame(correct_data)
        incorr_df   = pd.DataFrame(incorrect_data)
        corr_df.to_excel(f'excel/{self.cfg.lang}_{self.cfg.fold}_correct_output.xlsx', index=False)  # index=False: 인덱스를 엑셀에 포함하지 않음
        incorr_df.to_excel(f'excel/{self.cfg.lang}_{self.cfg.fold}_incorrect_output.xlsx', index=False)  # index=False: 인덱스를 엑셀에 포함하지 않음
        
        print(f"{self.cfg.lang} / {self.cfg.fold}_Answer acc : {(count_correct/len(self.test_dataset))*100:.2f}")
        print(f"{self.cfg.lang} / {self.cfg.fold}_head acc : {(head_correct/len(self.test_dataset))*100:.2f}")
        print(f"{self.cfg.lang} / {self.cfg.fold}_relation acc : {(relation_correct/len(self.test_dataset))*100:.2f}")
        print(f"{self.cfg.lang} / {self.cfg.fold}_tail acc : {(tail_correct/len(self.test_dataset))*100:.2f}")


    