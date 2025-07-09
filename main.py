import warnings
warnings.filterwarnings('ignore')

import kelip
from src.utils import *
from src.dataset import *
from src.train import *
from src.evaluation import *
from arguments import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu 
    set_seed(42)
    _, preprocess, tokenizer = kelip.build_model('ViT-B/32')

    if args.action == 'train':
        train_data, valid_data, triple_ans_list, triple_target_num, ans_list, ans_list_num = get_data(cfg)

        print('='*20, ' Train Phase ', '='*20)
        train_dataset   = KBVQA_Dataset(cfg, train_data, tokenizer, preprocess, ans_list, triple_ans_list)
        train_loader    = DataLoader(dataset=train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, pin_memory=True)
        
        valid_dataset   = KBVQA_Dataset(cfg, valid_data, tokenizer, preprocess, ans_list, triple_ans_list)
        valid_loader    = DataLoader(dataset=valid_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, pin_memory=True)
        
        trainer         = Trainer(cfg, tokenizer, train_loader, valid_loader, triple_ans_list, triple_target_num, ans_list, ans_list_num)
        trainer.train()

    else:
        test_data, triple_ans_list, triple_target_num, ans_list, ans_list_num = get_data(cfg)

        print('='*20, ' Test Phase ', '='*20)
        test_dataset = KBVQA_Dataset(cfg, test_data, tokenizer, preprocess, ans_list, triple_ans_list)
        tester       = Tester(cfg, tokenizer, test_dataset,  triple_ans_list, triple_target_num, ans_list, ans_list_num)
        tester.test()


if __name__ == "__main__":

    args    = get_args()
    cfg     = get_config(args.config_path)
    Config  = merge_config_and_args(cfg, args)
    main(Config)
