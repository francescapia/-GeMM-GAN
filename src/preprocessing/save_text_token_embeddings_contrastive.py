import torch
from torch import nn
from tqdm import tqdm
import json
from pathlib import Path
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import gc
import argparse
import numpy as np


# lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description='Save text embeddings of Clinical ModernBERT')
    parser.add_argument('--dataset_path', type=str, default='/CompanyDatasets/carlos00/all_tcga_100mb', help='Path to the dataset')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension for the text encoder')
    parser.add_argument('--model_path', type=str, default='/Experiments/carlos00/iciap/contrastive_model/emb_256/2025-05-20_21-24-42', help='Path to the model')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    emb_dim = args.emb_dim
    models_path = Path(args.model_path)

    with open(dataset_path / 'descriptions.json', 'r') as f:
        descriptions = json.load(f)
    
    image_embeddings = []
    case_ids = descriptions.keys()

    embeddings_path_bert = dataset_path / '..' / 'clinical_modernbert_embeddings'
    embeddings_path = dataset_path / '..' / f'text_embeddings_contrastive_{emb_dim}'
    embeddings_path.mkdir(parents=True, exist_ok=True)
    
    text_encoder = nn.Linear(768, emb_dim)
    text_encoder.load_state_dict(torch.load(str(models_path / 'text_encoder.pth')))
    text_encoder = text_encoder.cuda()
    text_encoder.eval()
    
    with torch.no_grad():
        for i, case_id in tqdm(enumerate(case_ids)):
            bert_embeddings = np.load(str(embeddings_path_bert / f"{case_id}.npy"))
            attention_mask = np.load(str(embeddings_path_bert / f"{case_id}_attention_mask.npy"))

            bert_embeddings = torch.tensor(bert_embeddings, dtype=torch.float32).cuda()
            embeddings = text_encoder(bert_embeddings).cpu().numpy()

            np.save(str(embeddings_path / f"{case_id}.npy"), embeddings)
            np.save(str(embeddings_path / f"{case_id}_attention_mask.npy"), attention_mask)
    
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()