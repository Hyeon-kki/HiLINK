import torch
import torch.nn.functional as F

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):

    image_embeddings    = image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings     = text_embeddings = F.normalize(text_embeddings, dim=-1)

    logits_per_image    = (image_embeddings @ text_embeddings.T) / temperature
    logits_per_text     = (text_embeddings @ image_embeddings.T) / temperature


    targets     = torch.arange(image_embeddings.size(0)).to(image_embeddings.device)

    loss_image  = F.cross_entropy(logits_per_image, targets)
    loss_text   = F.cross_entropy(logits_per_text, targets)

    loss        = (loss_image + loss_text) / 2
    return loss