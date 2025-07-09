import torch
from torch import nn

class PromptLearner(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        n_cls           = 1
        n_ctx           = 25
        self.model      = model
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype      = torch.float16
        self.ctx_dim    = 512
        self.tokenizer  = tokenizer

        ## shared parameter를 확인해야함. 24개의 vector에 대해서
        ## Frozen / non-Frozen 다 확인해보기
        
        print("Initializing a generic context")

        # shared token
        ctx_vectors = torch.empty(1, n_ctx, self.ctx_dim, dtype=self.dtype) # (1, n_ctx, 512)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors).to(self.device)  # to be optimized # (1, n_ctx, 512)

        self.prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        
        self.n_cls                  = n_cls
        self.n_ctx                  = n_ctx
        self.class_token_position   = 'end'

        # self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self, prompt_suffixs, batch_size):

        expanded_ctx        = self.ctx.expand(batch_size, -1, -1)
        prompts             = [self.prompt_prefix + " " + item for item in prompt_suffixs]
        tokenized_prompts   = self.tokenizer.encode(prompts).to(self.device)# (B, 77)

        with torch.no_grad():
            embedding = self.model.text_embedding(tokenized_prompts).to(self.device) # (B, 77, 512)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompt = torch.cat(
                [
                    prefix,           # (n_cls, 1, dim)
                    expanded_ctx,     # (n_cls, n_ctx, dim)
                    suffix,           # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len    = self.name_lens[i]
                prefix_i    = prefix[i : i + 1, :, :]
                class_i     = suffix[i : i + 1, :name_len, :]
                suffix_i    = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = self.ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = self.ctx[i : i + 1, half_n_ctx:, :]
                prompt      = torch.cat(
                            [
                                prefix_i,     # (1, 1, dim)
                                ctx_i_half1,  # (1, n_ctx//2, dim)
                                class_i,      # (1, name_len, dim)
                                ctx_i_half2,  # (1, n_ctx//2, dim)
                                suffix_i,     # (1, *, dim)
                            ],
                            dim=1,
                        )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i  = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i    = self.ctx[i : i + 1, :, :]
                prompt   = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompt, tokenized_prompts # 1x77x512