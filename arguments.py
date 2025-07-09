import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('action',       type=str, default='train',  help='Action')
    parser.add_argument("--lang",       type=str, required=True,    choices=['ko', 'en', 'bi'])
    parser.add_argument("--file_name",  type=str, required=False)
    parser.add_argument("--pt_name",  type=str, required=False)
    parser.add_argument('--config_path',type=str, required=False,   default='./config/base.yaml',  help='config yaml file')
    parser.add_argument('--gpu',        type=str, required=False,   default='0')
    parser.add_argument("--fold",       type=int, required=False,   default=1)
    parser.add_argument("--seed",       type=int, required=False,   default=42)
    parser.add_argument("--kge_data",   type=str, required=False,   default='all')
    parser.add_argument('--logging',    type=bool,default=False,    help='Logging option') 
    parser.add_argument("--kge_model",  type=str, required=False,   choices=['complex', 'convkb','TransE', 'TransH', 'TransR', 'TransD', 'TorusE', 'RESCAL', 'DistMult', 'HolE', 'ComplExModel', 'Analogy'], default='convkb')

    return parser.parse_args()
