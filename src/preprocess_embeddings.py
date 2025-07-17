import torch
from tqdm import tqdm
import openslide
import numpy as np
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import json
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from torchvision import transforms
from contrastive_model import UNI_FeatureExtractor, HuggingFaceTextEncoder
from transformers import AutoModel, AutoTokenizer
import gc
import threading
import concurrent.futures
import argparse


# lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path', type=str, default='/CompanyDatasets/carlos00/dataset_project', help='Path to the dataset')
    parser.add_argument('--model_path', type=str, default='/Experiments/carlos00/llm_project/contrastive_model/2025-04-27_16-41-14', help='Path to the model')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    return parser.parse_args()


def process_case(dataset_path, info, valid_tiles, image_encoder, transform):
    case_id = info['case_id']
    slide_path = dataset_path / 'tissue' / info['tissue_files'][0]
    patches = valid_tiles[case_id]['256']
    total_case_embeddings = np.empty((0, 128))
    
    total_patches = len(patches)
    for j in range(0, total_patches, 64):
        if j + 64 < len(patches):
            patches_subset = patches[j:j+64]
        else:
            patches_subset = patches[j:]
            
        case_embeddings = []
        for patch in patches_subset:
            image = openslide.OpenSlide(slide_path).read_region((int(patch[0]), int(patch[1])), 0, (int(patch[2]), int(patch[3]))).convert("RGB")
            if image.size[0] != 256 or image.size[1] != 256:
                new_image = Image.new("RGB", (256, 256), (255, 255, 255))
                new_image.paste(image, (0, 0))
                image = new_image
            image = transform(image)
            case_embeddings.append(image)
        
        case_embeddings = torch.stack(case_embeddings, dim=0)
        
        # with lock:
        with torch.no_grad():
            case_embeddings = image_encoder(case_embeddings.cuda()).cpu().numpy()
        
        total_case_embeddings = np.concatenate([total_case_embeddings, case_embeddings], axis=0)
        
    np.save(str(dataset_path / 'patch_embeddings' / f"{case_id}.npy"), total_case_embeddings)    
    return case_id, np.mean(total_case_embeddings, axis=0)


if __name__ == '__main__':
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    models_path = Path(args.model_path)

    with open(dataset_path / 'dataset_info.pkl', 'rb') as f:
        infos = pickle.load(f)['data_list']

    with open(dataset_path / 'valid_patches.json', 'r') as f:
        valid_tiles = json.load(f)
        
    with open(dataset_path / 'metainfos.pkl', 'rb') as f:
        metainfos = pickle.load(f)
        
    with open(dataset_path / 'descriptions.json', 'r') as f:
        descriptions = json.load(f)

    image_encoder = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    image_encoder = UNI_FeatureExtractor(image_encoder, output_dim=128)
    image_encoder.load_state_dict(torch.load(str(models_path / 'image_encoder.pth')))
    image_encoder = image_encoder.cuda()
    image_encoder.eval()
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    image_embeddings = []
    case_ids = []
    
    (dataset_path / 'patch_embeddings').mkdir(parents=True, exist_ok=True)
    
    for info in tqdm(infos):
        case_id, embedding = process_case(dataset_path, info, valid_tiles, image_encoder, transform)
        image_embeddings.append(embedding)
        case_ids.append(case_id)
    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    #     futures = [executor.submit(process_case, dataset_path, info, valid_tiles, image_encoder, transform) for info in infos]
        
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
    #         case_id, mean_embedding = future.result()
    #         mean_embeddings.append(mean_embedding)
    #         case_ids.append(case_id)
            
    image_embeddings = np.stack(image_embeddings, axis=0)
    pd.DataFrame(image_embeddings, index=case_ids).to_parquet(str(dataset_path / 'image_embeddings_contrastive.parquet'), index=True)
    
    del image_encoder
    gc.collect()
    torch.cuda.empty_cache()
    
    text_encoder = AutoModel.from_pretrained('Simonlee711/Clinical_ModernBERT')
    text_encoder = HuggingFaceTextEncoder(text_encoder, output_dim=128)
    text_encoder.load_state_dict(torch.load(str(models_path / 'text_encoder.pth')))
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
            # inputs = tokenizer(descr, return_tensors='pt', padding=True, truncation=True)
            outputs = text_encoder(input_ids, attention_mask)
            embeddings.append(outputs)
            
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    pd.DataFrame(embeddings, index=case_ids).to_parquet(str(dataset_path / 'text_embeddings_contrastive.parquet'), index=True)
    
    del tokenizer, text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    
    embeddings_global = (image_embeddings + embeddings) / 2
    pd.DataFrame(embeddings_global, index=case_ids).to_parquet(str(dataset_path / 'embeddings_contrastive.parquet'), index=True)