import torch
from tqdm import tqdm
import json
from pathlib import Path
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import gc
import argparse


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
    
    text_encoder = AutoModel.from_pretrained('Simonlee711/Clinical_ModernBERT')
    text_encoder = text_encoder.cuda()
    text_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained('Simonlee711/Clinical_ModernBERT')
    
    embeddings = []

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
            cls_output = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_output)
            
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    pd.DataFrame(embeddings, index=case_ids).to_parquet(str(dataset_path / 'clinical_modernbert_embeddings.parquet'), index=True)
    
    del tokenizer, text_encoder
    gc.collect()
    torch.cuda.empty_cache()