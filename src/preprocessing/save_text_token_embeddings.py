import torch
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_path = Path(args.dataset_path)
        
    with open(dataset_path / 'descriptions.json', 'r') as f:
        descriptions = json.load(f)
    
    image_embeddings = []
    case_ids = descriptions.keys()

    embeddings_path = dataset_path / '..' / 'clinical_modernbert_embeddings'
    embeddings_path.mkdir(parents=True, exist_ok=True)
    
    text_encoder = AutoModel.from_pretrained('Simonlee711/Clinical_ModernBERT')
    text_encoder = text_encoder.cuda()
    text_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained('Simonlee711/Clinical_ModernBERT')
    
    with torch.no_grad():
        for i, case_id in tqdm(enumerate(case_ids)):
            descr = descriptions[case_id]
            encoding = tokenizer(
                descr,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=300
            )
            input_ids = encoding['input_ids'].cuda()
            attention_mask = encoding['attention_mask'].cuda()
            outputs = text_encoder(input_ids, attention_mask)
            embeddings = outputs.last_hidden_state
            np.save(str(embeddings_path / f"{case_id}.npy"), embeddings.cpu().numpy())
            np.save(str(embeddings_path / f"{case_id}_attention_mask.npy"), attention_mask.cpu().numpy())
    
    del tokenizer, text_encoder
    gc.collect()
    torch.cuda.empty_cache()