import yaml
from box import Box
import torch 
import pandas as pd
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import torch
import pickle
import os
from torchkge.models import ConvKBModel
from torchkge.models import ComplExModel
from torchkge.models.translation import TransEModel, TransHModel, TransRModel, TransDModel,TorusEModel
from torchkge.models.bilinear import RESCALModel, DistMultModel, HolEModel, AnalogyModel

import torchvision.transforms as transforms
from transformers import AutoTokenizer
import transformers
import torchvision.models as models
import argparse
import neptune

def set_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def neptune_load(cfg):

    selected_params = { "backbone"  : cfg.train.backbone,
                        "epochs"     : cfg.train.epoch,
                        "batch_size" : cfg.train.batch_size,
                        "lr"         : cfg.train.lr,
                        "drop_out"   : cfg.train.drop_out,
                      }

    run = neptune.init_run(
    project="HiLINK",
    name=cfg.train.experiment,
    api_token="neptune_api_token",  # replace with your own Neptune API token
    )
    run["sys/tags"].add(cfg.tag+cfg.lang+'_'+str(cfg.fold))
    return run
        
def merge_config_and_args(config, args):
    var_args = vars(args)
    for key, value in var_args.items():
        if value is not None:
            config[key] = value
    return config

def get_config(path):
    # load
    with open(path, 'r', encoding='utf8') as file:
        config = yaml.safe_load(file)

    # Hierarchical box
    cfg = Box(config)
    return cfg

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val    = 0
        self.avg    = 0
        self.sum    = 0
        self.count  = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum    += val * n
        self.count  += n
        self.avg    = self.sum / self.count

def get_triple_list(data):
    h_list = sorted(list(set(data['h'])))
    r_list = sorted(list(set(data['r'])))
    t_list = sorted(list(set(data['t'])))

    triple_ans_list = {"h":h_list, "r":r_list, "t":t_list}
    triple_target_num = {"h":len(h_list), "r":len(r_list), "t":len(t_list)}

    return triple_ans_list, triple_target_num

def get_data(cfg):

    if cfg.lang == 'bi':
        data_ko = pd.read_csv(os.path.join(cfg.path.train_path, "BOKVQA_data_ko.csv"))
        data_en = pd.read_csv(os.path.join(cfg.path.train_path, "BOKVQA_data_en.csv"))
        data    = pd.concat([data_ko, data_en])
        data    = data.sort_values(by="img_path").reset_index(drop=True)
    else:
        data    = pd.read_csv(os.path.join(cfg.path.train_path, f'BOKVQA_data_{cfg.lang}.csv'))
    
    train_data  = data[data[f'fold']!=cfg.fold].reset_index(drop=True)
    valid_data  = data[data[f'fold']==cfg.fold].reset_index(drop=True)
    
    train_ans_list                                  = sorted(list(pd.concat([train_data, valid_data], axis=0)['answer'].unique()))
    train_triple_ans_list, train_triple_target_num  = get_triple_list(data)

    if cfg.action == 'train':
        return train_data, valid_data, train_triple_ans_list, train_triple_target_num, train_ans_list, len(train_ans_list)
    
    else:
        if cfg.lang == 'bi':
            data_ko     = pd.read_csv(os.path.join(cfg.path.test_path, "BOKVQA_data_test_ko.csv"))
            data_en     = pd.read_csv(os.path.join(cfg.path.test_path, "BOKVQA_data_test_en.csv"))
            test_data   = pd.concat([data_ko, data_en])
        else:
            test_data   = pd.read_csv(os.path.join(cfg.path.test_path, f'BOKVQA_data_test_{cfg.lang}.csv'))

        test_data       = test_data.reset_index(drop= True)
        
        return test_data, train_triple_ans_list, train_triple_target_num, train_ans_list, len(train_ans_list)
    
def get_transform():

    train_transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform


def get_embedded_vec(batch_size, heads, rels, tails, emb_entity_ , emb_rel_, kg):
    head_emb = emb_entity_[[kg.ent2ix[i] for i in heads]]
    rel_emb  = emb_rel_[[kg.rel2ix[i] for i in rels]]
    tail_emb = emb_entity_[[kg.ent2ix[i] for i in tails]]
    emb_list = []
    for i in range(batch_size):
        a = torch.tensor([head_emb[i], rel_emb[i], tail_emb[i]], dtype=torch.float).reshape(-1,768)
        emb_list.append(a)
    
    return torch.cat(emb_list, 0)
        
def save_results(version, train_loss, train_acc, valid_loss, valid_acc):
    PATH = f"./saved_results/{version}"
    with open(os.path.join(PATH, "train_loss.pkl"), 'wb') as f:
        pickle.dump(train_loss, f)

    with open(os.path.join(PATH, "train_acc.pkl"), 'wb') as f:
        pickle.dump(train_acc, f)

    with open(os.path.join(PATH, "valid_loss.pkl"), 'wb') as f:
        pickle.dump(valid_loss, f)

    with open(os.path.join(PATH, "valid_acc.pkl"), 'wb') as f:
        pickle.dump(valid_acc, f)

def get_kelip_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    return tokenizer

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    return tokenizer


def get_answer_dict():
    ko_data = pd.read_csv(os.path.join(data_dir, "BOKVQA_data_ko.csv"))
    en_data = pd.read_csv(os.path.join(data_dir, "BOKVQA_data_en.csv"))
    data = pd.concat([ko_data, en_data])

    ko_data_ = ko_data[['img_path', 'answer']].rename(columns={'answer': 'answer_ko'})
    en_data_ = en_data[['img_path', 'answer']].rename(columns={'answer': 'answer_en'})

    merged_data = pd.merge(ko_data_, en_data_, on='img_path')
    k2e_answer_dict = pd.Series(merged_data.answer_en.values,index=merged_data.answer_ko).to_dict()
    e2k_answer_dict = pd.Series(merged_data.answer_ko.values,index=merged_data.answer_en).to_dict()

    return k2e_answer_dict, e2k_answer_dict

def get_num_workers():
    return 5

def get_batch_size():
    return 128

def get_seed():
    return 56

def get_language_model():
    return transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base', output_attentions=True)

def get_image_model():
    return models.resnet50(pretrained=True)

def loss_meter():
    
    loss_dic = {
        "train"     :  AverageMeter(),
        "answer"    : AverageMeter(),
        "head"      : AverageMeter(),
        "relation"  : AverageMeter(),
        "tail"      : AverageMeter(),
    }
    return loss_dic

def acc_meter():

    acc_dic = {
        "answer"    : AverageMeter(),
        "hrt"       : AverageMeter(),
        "head"      : AverageMeter(),
        "relation"  : AverageMeter(),
        "tail"      : AverageMeter(), 
    }
    return acc_dic

# clear
def get_KGE(config, kge_data='all', kge_model='complex'):
    kge_dict = {
        'TransE' : TransEModel,
        'TransH' : TransHModel,
        'TransR' : TransRModel,
        'TransD' : TransDModel,
        'TorusE' : TorusEModel,
        'RESCAL' : RESCALModel,
        'DistMult' : DistMultModel,
        'HolE' : HolEModel,
        'complex' : ComplExModel,
        'convkb' : ConvKBModel,
        'Analogy' : AnalogyModel
    }

    with open(os.path.join(config.path.kge_path, f'kge_save/{kge_model}_{config.kge.kge_n_iter}_{config.kge.kge_lr}_{config.kge.kge_batch}_{config.kge.kge_margin}_{kge_data}_config.pkl'), 'rb') as f:
        kge_config = pickle.load(f)

    with open(os.path.join(config.path.kge_path, f'kge_save/{kge_model}_{config.kge.kge_n_iter}_{config.kge.kge_lr}_{config.kge.kge_batch}_{config.kge.kge_margin}_{kge_data}_kg.pkl'), 'rb') as f:
        kg = pickle.load(f)

    model = kge_dict[kge_model]

    if kge_model == 'convkb':
        KGEModel = model(kge_config['emb_dim'],
                                      3,
                                      kge_config['n_ent'],
                                      kge_config['n_rel']
                                    )
        
    elif kge_model in ["TransR", "TransD"]:
        KGEModel = model(      
                    kge_config['emb_dim'],
                    kge_config['emb_dim'],
                    kge_config['n_ent'],
                    kge_config['n_rel']
                )
    elif kge_model in ["TorusE"]:    
         KGEModel = model(      
                    kge_config['emb_dim'],
                    kge_config['n_ent'],
                    kge_config['n_rel'],
                    dissimilarity_type = 'torus_L2'
                    )
        
    else:
        KGEModel = model(kge_config['emb_dim'],
                         kge_config['n_ent'],
                         kge_config['n_rel']
                         )
      
    KGEModel.load_state_dict(torch.load(os.path.join(config.path.kge_path, f'kge_save/{kge_model}_{config.kge.kge_n_iter}_{config.kge.kge_lr}_{config.kge.kge_batch}_{config.kge.kge_margin}_{kge_data}.pt')))
            
    emb_entity_ = KGEModel.get_embeddings()[0].detach().cpu().numpy()
    emb_rel_ = KGEModel.get_embeddings()[1].detach().cpu().numpy()

    return KGEModel, kg, emb_entity_, emb_rel_    
