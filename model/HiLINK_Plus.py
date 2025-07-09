import kelip
from torch import nn
from model.prompt_learner import *

class HiLINK_Plus(nn.Module):  # Fusion method: Element-wise Multiplication, late Fusion
    def __init__(self, cfg, num_target_gold, num_target_triple, dim_i=768, dim_h=1024):
        super(HiLINK_Plus, self).__init__()
        
        self.cfg = cfg
        self.model, self.preprocess, self.tokenizer = kelip.build_model('ViT-B/32')
        self.prompt_learner = PromptLearner(self.model, self.tokenizer)
        self.relu           = nn.ReLU(inplace=False)
        self.drop           = nn.Dropout(0.2)
        self.h_layer_norm   = nn.LayerNorm(dim_h)

        ## classfier layer
        self.linear_gold    = nn.Linear(dim_h, num_target_gold)

        self.linear_t       = nn.Sequential(
                                nn.Linear(dim_h, 2*dim_h),
                                nn.LayerNorm(2*dim_h),
                                nn.ReLU(),
                                nn.Linear(2*dim_h, 3*dim_h),
                                nn.LayerNorm(3*dim_h),
                                nn.ReLU(),
                                nn.Linear(3*dim_h, num_target_triple['t']),
                              )

        self.linear_h       = nn.Sequential(
                                nn.Linear(dim_h, 2*dim_h),
                                nn.LayerNorm(2*dim_h),
                                nn.ReLU(),
                                nn.Linear(2*dim_h, 3*dim_h),
                                nn.LayerNorm(3*dim_h),
                                nn.ReLU(),
                                nn.Linear(3*dim_h, num_target_triple['h']),
                              )
        
        self.classifier_linear_q    = nn.Linear(512, dim_h)
        self.classifier_linear_iq   = nn.Linear(1024, dim_h)
        self.classifier_linear_ipq  = nn.Linear(1024, dim_h)

        self.projection_linear_i    = nn.Linear(512, 768)
        self.projection_linear_q    = nn.Linear(512, 768)
        self.projection_linear_pq   = nn.Linear(512, 768)
        
        self.modality_layer_norm    = nn.LayerNorm(768)

        ## Knowledge FiLM
        self.h_film_gen = FiLMGenerator(num_target_triple['h'], 3*dim_h)
        self.t_film_gen = FiLMGenerator(num_target_triple['t'], 3*dim_h)
        self.r_film_gen = FiLMGenerator(num_target_triple['r'], 3*dim_h)
        
        # self.h_film = FiLM(dim_h)
        self.r_film = r_FiLM(dim_h, num_target_triple)
        self.h_film = h_FiLM(dim_h, num_target_triple)
        self.t_film = t_FiLM(dim_h, num_target_triple)
    
    def forward(self, learnable_text, questions, image, tokenized_prompts=0, get_answer=True):

        ## Image Feature
        with torch.no_grad():
            image_f = self.model.encode_image(image) 
        query_f     = self.model.encode_text(questions) 
        fusion_f    = torch.cat((image_f, query_f), dim=-1) 

        out_query   = self.drop(self.relu(self.h_layer_norm(self.classifier_linear_q(query_f))))
        out_imgqry  = self.drop(self.relu(self.h_layer_norm(self.classifier_linear_iq(fusion_f))))
        
        f_out_h     = self.linear_h(out_imgqry)
        
        h_gamma, h_beta = self.h_film_gen(f_out_h)
        f_out_r         = self.r_film(out_query, h_gamma, h_beta)
        
        ## Tail Prediction
        r_gamma, r_beta = self.r_film_gen(f_out_r)
        f_out_t         = self.t_film(out_query, r_gamma, r_beta)
        
        ## Head Prediction
        b_out_t = self.linear_t(out_query)
        
        t_gamma, t_beta = self.t_film_gen(b_out_t)
        b_out_r         = self.r_film(out_query, t_gamma, t_beta)
        
        ## Tail Prediction
        r_gamma, r_beta = self.r_film_gen(b_out_r)
        b_out_h         = self.h_film(out_imgqry, r_gamma, r_beta)

        out_h           = (f_out_h + b_out_h)/2
        out_r           = (f_out_r + b_out_r)/2
        out_t           = (f_out_t + b_out_t)/2

        if not get_answer:
            return out_h, out_r, out_t
        
        ## Answer Prediction
        prompt_f = self.model.encode_text_learnable(learnable_text, tokenized_prompts) 

        # ## Image-Prompt text Fusion
        pfusion_f   = torch.cat((image_f, prompt_f), dim=-1) 

        out_uni     = self.drop(self.relu(self.h_layer_norm(self.classifier_linear_ipq(pfusion_f))))
        out_gold    = self.linear_gold(out_uni)
        return out_gold, out_h, out_r, out_t
    
class FiLMGenerator(nn.Module):
    def __init__(self, text_dim, feature_dim):
        super(FiLMGenerator, self).__init__()
        self.gamma_fc = nn.Sequential(
            nn.Linear(text_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2)
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(text_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2)
        )

    def forward(self, text_features):
        gamma   = self.gamma_fc(text_features)  
        beta    = self.beta_fc(text_features)  
        return gamma, beta

class r_FiLM(nn.Module):
    def __init__(self, feature_dim, num_target_triple):
        super(r_FiLM, self).__init__()
        self.feature_dim = feature_dim
        self.sequential  = nn.Sequential(
            nn.Linear(feature_dim, 2*feature_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(2*feature_dim),
            nn.Linear(2*feature_dim, 3*feature_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(3*feature_dim),
        )
        self.linear_r = nn.Linear(3*feature_dim, num_target_triple['r'])

    def forward(self, features, gamma, beta):
        out_features        = self.sequential(features)
        modulated_features  = gamma * out_features + beta + out_features
        modulated_features  = self.linear_r(modulated_features)

        return modulated_features
    
class h_FiLM(nn.Module):
    def __init__(self, feature_dim, num_target_triple):
        super(h_FiLM, self).__init__()
        self.feature_dim = feature_dim
        self.sequential  = nn.Sequential(
            nn.Linear(feature_dim, 2*feature_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(2*feature_dim),
            nn.Linear(2*feature_dim, 3*feature_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(3*feature_dim),
        )
        self.linear_t = nn.Linear(3*feature_dim, num_target_triple['h'])

    def forward(self, features, gamma, beta):
        out_features        = self.sequential(features)
        modulated_features  = gamma * out_features + beta + out_features
        modulated_features  = self.linear_t(modulated_features)
        return modulated_features
    
class t_FiLM(nn.Module):
    def __init__(self, feature_dim, num_target_triple):
        super(t_FiLM, self).__init__()
        self.feature_dim = feature_dim
        self.sequential  = nn.Sequential(
            nn.Linear(feature_dim, 2*feature_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(2*feature_dim),
            nn.Linear(2*feature_dim, 3*feature_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(3*feature_dim),
        )
        self.linear_t = nn.Linear(3*feature_dim, num_target_triple['t'])

    def forward(self, features, gamma, beta):
        out_features        = self.sequential(features)
        modulated_features  = gamma * out_features + beta + out_features
        modulated_features  = self.linear_t(modulated_features)
        return modulated_features