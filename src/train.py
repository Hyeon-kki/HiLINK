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
from model.prompt_learner import *
import logging
from src.metric import *

class Trainer:

    def __init__(self, cfg, tokenizer, train_loader, valid_loader, triple_ans_list, triple_target_num, ans_list, ans_list_num):
        self.cfg            = cfg
        self.model          = HiLINK_Plus(cfg, ans_list_num, triple_target_num)
        self.tokenizer      = tokenizer
        self.optimizer      = optim.AdamW(self.model.parameters(), lr=self.cfg.train.lr)
        self.criterion      = nn.CrossEntropyLoss()


        ## Data loader
        self.train_loader       = train_loader
        self.valid_loader       = valid_loader

        ## Answer list
        self.triple_ans_list    = triple_ans_list
        self.triple_target_num  = triple_target_num
        self.ans_list           = ans_list
        self.ans_list_num       = ans_list_num

        self.cur_epoch  = 0
        self.cur_step   = 0
        self.total_step = len(self.train_loader) * self.cfg.train.epoch

        # logging
        if cfg.logging:
            self.run = neptune_load(self.cfg)

        self.device = self._get_device()

        log_file_name = cfg.train.experiment + '_log_' + cfg.lang +'_'+ str(cfg.fold)+'.log'
        log_file_path = os.path.join(cfg.path.log_path, log_file_name)
        logging.basicConfig(filename=log_file_path, 
                            filemode='w', 
                            level=logging.INFO, 
                            format='%(asctime)s - %(message)s')

    def _save_log(self, msg):
        with open(f'{self.cfg.checkpoint}/log.txt', 'a') as f:
            f.write(msg)

    def train(self,):
        best_epoch     = 0
        best_acc       = 0
        best_acc_model = None

        print("======================= START TRAINING =======================")
        print(f"model   : PLINK")
        print(f"lang    : {self.cfg.lang}")
        print(f"version : {self.cfg.train.experiment}")

        for epoch in range(1, self.cfg.train.epoch+1):

            ## Train phase
            train_loss_dic = loss_meter()
            train_acc_dic  = acc_meter()
            
            self.model = self.model.to(self.device).float()
            self.model.train()

            train_loss_dic, train_acc_dic = self._run_epoch(self.train_loader, train_loss_dic, train_acc_dic)
            
            ## Valid phase
            valid_loss_dic = loss_meter()
            valid_acc_dic  = acc_meter()

            self.model.eval()
            valid_loss_dic, valid_acc_dic = self._run_epoch(self.valid_loader, valid_loss_dic, valid_acc_dic, valid = True)

            if valid_acc_dic['answer'].avg*100 > best_acc:
                best_acc = valid_acc_dic['answer'].avg*100
                best_epoch = epoch
                best_acc_model = deepcopy(self.model.state_dict())

            if self.cfg.logging == True:
                self.run['cur epoch'].append(epoch)
                self.run['train loss'].append(train_loss_dic['train'].avg)
                self.run['train answer loss'].append(train_loss_dic['answer'].avg*100)
                self.run['train head loss'].append(train_loss_dic['head'].avg*100)
                self.run['train relation loss'].append(train_loss_dic['relation'].avg*100)
                self.run['train tail loss'].append(train_loss_dic['tail'].avg*100)

                self.run['val loss'].append(valid_loss_dic['train'].avg)
                self.run['val answer loss'].append(valid_loss_dic['answer'].avg*100)
                self.run['val head loss'].append(valid_loss_dic['head'].avg*100)
                self.run['val relation loss'].append(valid_loss_dic['relation'].avg*100)
                self.run['val tail loss'].append(valid_loss_dic['tail'].avg*100)

                self.run['T answer acc'].append(train_acc_dic['answer'].avg*100)
                self.run['T head acc'].append(train_acc_dic['head'].avg*100)
                self.run['T relation acc'].append(train_acc_dic['relation'].avg*100)
                self.run['T tail acc'].append(train_acc_dic['tail'].avg*100)

                self.run['V answer acc'].append(valid_acc_dic['answer'].avg*100)
                self.run['V head acc'].append(valid_acc_dic['head'].avg*100)
                self.run['V relation acc'].append(valid_acc_dic['relation'].avg*100)
                self.run['V tail acc'].append(valid_acc_dic['tail'].avg*100)
                
            print(f"BEST EPOCH    : {best_epoch:2.2f}")
            print(f"BEST VALID ACC: {best_acc:2.2f}")

            self.cur_epoch += 1
        
        self._save_checkpoint(best_acc_model, best_acc)
        best_str = f"BEST EPOCH : {best_epoch:2.2f}, BEST VALID ACC : {best_acc:2.2f}"
        logging.info(best_str)
        if self.cfg.logging:
            self.run.stop()
    
    def _save_checkpoint(self, best_acc_model, best_acc):

        save_path = os.path.join(self.cfg.path.save_path, self.cfg.train.experiment, self.cfg.lang)
        os.makedirs(save_path, exist_ok=True)
        best_acc_model = torch.save(best_acc_model, f"{save_path}/{self.cfg.tag}_{self.cfg.lang}_{self.cfg.fold}_{best_acc:.2f}.pt")

    def _get_acc(self, outputs, labels ,batch_size):

        preds          = torch.argmax(outputs, dim=1)
        count_corrects = labels.eq(preds).sum().item()
        acc            = count_corrects/batch_size
        return acc
    
    def _get_device(self,):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _run_epoch(self, loader, loss_dic, acc_dic, valid = False):

        iterator = tqdm(enumerate(loader), total=len(loader), unit='Iter')

        for idx, batch in iterator:

            batch_size  = batch['A'].size(0)
            answers     = batch['A'].to(self.device)
            questions   = batch['Q'].squeeze(1).to(self.device)
            nTQ         = batch['nTQ']  
            imgs        = batch['img'].to(self.device)  

            h_label     = batch['h_label'].squeeze().to(self.device)
            r_label     = batch['r_label'].squeeze().to(self.device)
            t_label     = batch['t_label'].squeeze().to(self.device)
            
            knowledge_questions = ''

            with torch.no_grad():
                h_out, r_out, t_out = self.model(knowledge_questions, questions, imgs, get_answer=False) 

                h_pred      = torch.argmax(h_out, dim=1)
                r_pred      = torch.argmax(r_out, dim=1)
                t_pred      = torch.argmax(t_out, dim=1)

                ## Valid 
                head_pred   = [self.triple_ans_list['h'][i] for i in h_pred]
                rel_pred    = [self.triple_ans_list['r'][i] for i in r_pred]
                tail_pred   = [self.triple_ans_list['t'][i] for i in t_pred]

                ## Teaching Force
                head        = [self.triple_ans_list['h'][i] for i in h_label]
                rel         = [self.triple_ans_list['r'][i] for i in r_label]
                tail        = [self.triple_ans_list['t'][i] for i in t_label]

            ## Prompt design
            if valid:
                knowledge_questions = [a + ' ' + b + ' ' + c + ' ' + d + '.' for a, b, c, d in zip(nTQ, head_pred, rel_pred, tail_pred)]
            else:
                knowledge_questions = [a + ' ' + b + ' ' + c + ' ' + d + '.' for a, b, c, d in zip(nTQ, head, rel, tail)]

            ## Prompt Learnable
            knowledge_prompt, tokenized_prompts = self.model.prompt_learner(knowledge_questions, batch_size)
            knowledge_prompt                    = knowledge_prompt.squeeze(1).to(self.device)
            answer_out, h_out, r_out, t_out     = self.model(knowledge_prompt, questions, imgs, tokenized_prompts, get_answer=True)
            
            ## Loss
            # con_loss = contrastive_loss(ko_query_f, en_query_f)
            loss_a, loss_h, loss_r, loss_t      = self.criterion(answer_out, answers), self.criterion(h_out, h_label), self.criterion(r_out, r_label), self.criterion(t_out, t_label)
            total_loss                          = loss_a + loss_h + loss_r + loss_t 
            if not valid:
                self.optimizer.zero_grad()
                total_loss.backward(total_loss)
                self.optimizer.step()

            loss_list = [total_loss, loss_a, loss_h, loss_r, loss_t]
            acc_list  = []

            acc_list.append(self._get_acc(answer_out, answers, batch_size))
            acc_list.append(self._get_acc(h_out, h_label, batch_size))
            acc_list.append(self._get_acc(r_out, r_label, batch_size))
            acc_list.append(self._get_acc(t_out, t_label, batch_size))

            hrt_count_correct   = (torch.stack((h_label, r_label, t_label)).eq(torch.stack((h_pred, r_pred, t_pred))).float().sum(dim=0) == 3).sum().item()
            hrt_acc             = hrt_count_correct/batch_size
            acc_list.append(hrt_acc)

            result_loss_dic, result_acc_dic = self._metric_update(loss_list, acc_list, loss_dic, acc_dic, batch_size)

            if not valid:
                log_message = (f"[{self.cur_epoch:2}/{self.cfg.train.epoch}]  [TRAIN LOSS: {result_loss_dic['train'].avg:.4f}]  "
                            f"[TRAIN ACC: {result_acc_dic['answer'].avg*100:2.2f}]  [Head : {result_acc_dic['head'].avg*100:2.2f}]  "
                            f"[Rel : {result_acc_dic['relation'].avg*100:2.2f}]  [Tail : {result_acc_dic['tail'].avg*100:2.2f}]  "
                            f"[ALL : {acc_dic['hrt'].avg*100:2.2f}]")
            else:
                log_message = (f"[{self.cur_epoch:2}/{self.cfg.train.epoch}]  [Valid LOSS: {result_loss_dic['train'].avg:.4f}]  "
                            f"[Valid ACC: {result_acc_dic['answer'].avg*100:2.2f}]  [Head : {result_acc_dic['head'].avg*100:2.2f}]  "
                            f"[Rel : {result_acc_dic['relation'].avg*100:2.2f}]  [Tail : {result_acc_dic['tail'].avg*100:2.2f}]  "
                            f"[ALL : {acc_dic['hrt'].avg*100:2.2f}]")
            
            iterator.set_description(log_message)
        logging.info(log_message)

        return result_loss_dic, result_acc_dic
    
    def _metric_update(self, loss_list, acc_list, loss_dic, acc_dic, batch_size):
        
        loss_dic["train"].update(loss_list[0].item(), batch_size)
        loss_dic["answer"].update(loss_list[1].item(), batch_size)
        loss_dic["head"].update(loss_list[2].item(), batch_size)
        loss_dic["relation"].update(loss_list[3].item(), batch_size)
        loss_dic["tail"].update(loss_list[4].item(), batch_size)

        acc_dic["answer"].update(acc_list[0], batch_size)
        acc_dic["head"].update(acc_list[1], batch_size)
        acc_dic["relation"].update(acc_list[2], batch_size)
        acc_dic["tail"].update(acc_list[3], batch_size)
        acc_dic["hrt"].update(acc_list[4], batch_size)

        return loss_dic, acc_dic

