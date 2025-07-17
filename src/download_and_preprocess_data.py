import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import pickle
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import logging
import os
import gc
import timm
from torchvision import transforms
import torch
import torch.nn as nn
import openslide
import numpy as np
from PIL import Image
import queue
import threading
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import time
import json

from dataset.utils import (
    map_gdc_file, 
    get_metainfo_by_case_id, 
    download_file_by_id, 
    clean_rna_seq, 
    clean_tissue_slides,
    load_rna_seq_dataframe,
    save_case_id_to_slide_file_dataframe
)
from preprocessing.patch_preprocessing import extract_tiles


tissue_manifests = [
    '~/LLM_Project_2025/data/gdc_manifest.2025-04-18.223014_tissue_normal.txt',
    '~/LLM_Project_2025/data/gdc_manifest.2025-04-18.223200_tissue_tumor_unknown_specimen.txt',
    '~/LLM_Project_2025/data/gdc_manifest.2025-04-18.223419_tissue_tumor_solid_tissue_specimen_ffpe_oct_preservation.txt',
    '~/LLM_Project_2025/data/gdc_manifest.2025-04-18.223419_tissue_tumor_solid_tissue_specimen_unknown_preservation.txt'
]

rna_seq_manifests = [
    '~/LLM_Project_2025/data/gdc_manifest.2025-04-18.221417_rna_seq_part1.txt',
    '~/LLM_Project_2025/data/gdc_manifest.2025-04-18.221646_rna_seq_part2.txt',
]
progress_lock = threading.Lock()
model_lock = threading.Lock()
tiles_info_lock = threading.Lock()


def get_cases_info_from_file_ids(files_info_df: pd.DataFrame) -> pd.DataFrame:
    results = []

    # 100 is the maximum number of file IDs that can be sent in a single request
    num_files = files_info_df.shape[0]
    total_batches = (num_files // 100) + (1 if num_files % 100 != 0 else 0)
    for row in tqdm(range(total_batches), desc='Obtaining case IDs', unit='batch'):
        start_row_index = row * 100
        end_row_index = start_row_index + 100
        batch_file_ids = files_info_df.iloc[start_row_index:end_row_index].index.values.tolist()
        results.append(map_gdc_file(batch_file_ids))
                
    cases_df = pd.concat(results, axis=0)
    return cases_df


def download_files_parallel(files_df: pd.DataFrame, out_dir: Path, max_workers: int = 5,
                            max_retries: int = 4) -> List[bool]:
    out_dir.mkdir(exist_ok=True)
    results = []
    
    file_ids = files_df['file_id'].tolist()
    file_names = files_df['file_name'].tolist()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file_by_id, file_id, out_dir / file_name, max_retries) 
                   for file_id, file_name in zip(file_ids, file_names)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading files"):
            results.append(future.result())

    return results

def download_worker(file_info, output_dir, download_queue, max_retries, stop_event, progress_bar):
    case_id, file_id, file_name = file_info
    tmp_file_name = f"{case_id}_{file_name}"
    tmp_file_path = output_dir / tmp_file_name     

    if os.path.exists(tmp_file_path) or download_file_by_id(file_id, tmp_file_path, max_retries):
        download_queue.put((case_id, file_id, file_name, tmp_file_path))
    else:
        logging.info(f"Download failed for {tmp_file_name}")
        with progress_lock:
            progress_bar.update(1)
    
def otsu_mask_skimage(slide, level=6):
    thumbnail = np.array(slide.read_region((0, 0), 0, slide.level_dimensions[level]))[..., :3]
    
    gray = rgb2gray(thumbnail)
    thresh = threshold_otsu(gray)
    mask = gray < thresh

    return mask.astype(np.uint8), thumbnail

def preprocessing_worker(download_queue, preprocess_queue, stop_event, tiles_info, progress_bar):
    while not stop_event.is_set():
        try:
            case_id, file_id, file_name, slide_path = download_queue.get(timeout=5)
        except queue.Empty:
            continue

        try:
            slide = openslide.OpenSlide(slide_path)
            mask, slide_np = otsu_mask_skimage(slide, 0)
            tiles = extract_tiles(slide, mask, tile_size=256, background_thresh=0.8, mask_level=0)
            slide.close()
            
            with tiles_info_lock:
                tiles_info[case_id] = {256: tiles}

            preprocess_queue.put((case_id, slide_np, tiles))
        except Exception as e:
            logging.error(f"Error in preprocessing for {slide_path}: {e}")
            with progress_lock:
                progress_bar.update(1)
            continue
        finally:
            download_queue.task_done()
            os.remove(str(slide_path))

def embedding_worker(preprocess_queue, model, transforms, out_dir, embedding_size, progress_bar, stop_event):    
    while not stop_event.is_set():
        try:
            case_id, slide_np, tiles = preprocess_queue.get(timeout=5)
        except queue.Empty:
            continue

        try:
            if not os.path.exists(out_dir / f"{case_id}.npy"):
                total_case_embeddings = np.empty((0, embedding_size))
                for i in range(0, len(tiles), 64):
                    batch = tiles[i:i+64]
                    images = []

                    for patch in batch:
                        image = Image.fromarray(slide_np[patch[1]:patch[1]+patch[3], patch[0]:patch[0]+patch[2]])
                        if image.size != (256, 256):
                            new_img = Image.new("RGB", (256, 256), (255, 255, 255))
                            new_img.paste(image, (0, 0))
                            image = new_img
                        images.append(transforms(image))

                    batch_tensor = torch.stack(images)
                    with model_lock:
                        with torch.no_grad():
                            embeddings = model(batch_tensor.cuda()).cpu().numpy()
                    total_case_embeddings = np.concatenate([total_case_embeddings, embeddings], axis=0)

                np.save(str(out_dir / f"{case_id}.npy"), total_case_embeddings)

        except Exception as e:
            logging.error(f"Error in embedding for case {case_id}: {e}")
        finally:
            del slide_np, total_case_embeddings
            gc.collect()
            torch.cuda.empty_cache()
            
            with progress_lock:
                progress_bar.update(1)
            preprocess_queue.task_done()

def download_and_preprocess_tissues(model: torch.nn.Module, transforms, files_df: pd.DataFrame,
                                    out_dir: Path, max_workers: int = 2, max_retries: int = 10,
                                    embedding_size: int = 1024) -> List[bool]:
    
    out_dir.mkdir(exist_ok=True)
    
    download_queue = queue.Queue()
    preprocess_queue = queue.Queue()
    stop_event = threading.Event()

    file_infos = list(zip(files_df['case_id'], files_df['file_id'], files_df['file_name']))
    total_cases = len(file_infos)
    
    progress_bar = tqdm(total=len(file_infos), desc="Embedding progress", position=0)
    
    def monitor_progress():
        while not stop_event.is_set():
            if progress_bar.n >= total_cases:
                stop_event.set()
                break
            time.sleep(0.5)
            
    monitor = threading.Thread(target=monitor_progress)
    monitor.start()
    tiles_info = {}

    try:
        with ThreadPoolExecutor(max_workers=min(max_workers * 2, len(file_infos))) as download_pool, \
             ThreadPoolExecutor(max_workers=min(max_workers // 2, len(file_infos))) as preprocess_pool, \
             ThreadPoolExecutor(max_workers=min(max_workers // 2, len(file_infos))) as embedding_pool:

            for _ in range(max_workers // 2):
                embedding_pool.submit(embedding_worker, preprocess_queue, model, transforms, out_dir, embedding_size, progress_bar, stop_event)

            for _ in range(max_workers // 2):
                preprocess_pool.submit(preprocessing_worker, download_queue, preprocess_queue, stop_event, tiles_info, progress_bar)

            for file_info in file_infos:
                download_pool.submit(download_worker, file_info, out_dir, download_queue, max_retries, stop_event, progress_bar)

            download_pool.shutdown(wait=True)
            download_queue.join()
            preprocess_queue.join()

            embedding_pool.shutdown(wait=True)
            preprocess_pool.shutdown(wait=True)

    except KeyboardInterrupt:
        logging.warning("Interrupted. Shutting down...")
    finally:
        progress_bar.close()
        stop_event.set()
        monitor.join()
        
    with open(out_dir / '../valid_patches.json', 'w') as f:
        json.dump(tiles_info, f, indent=4)

    return True


def build_dataset_info(tissue_cases_df: pd.DataFrame, rna_seq_cases_df: pd.DataFrame) -> pd.DataFrame:
    dataset_info = {
        'metainfo': {
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': '1.0',
            'num_cases': tissue_cases_df['case_id'].unique().flatten().shape[0],
        },
        'data_list': []
    }
    
    for case_id in tqdm(tissue_cases_df['case_id'].unique(), desc='Building dataset info', unit='case'):
        case_info = {
            'case_id': case_id,
            'tissue_files': tissue_cases_df[tissue_cases_df['case_id'] == case_id]['file_name'].tolist(),
            'rna_seq_files': rna_seq_cases_df[rna_seq_cases_df['case_id'] == case_id]['file_name'].tolist(),
        }
        dataset_info['data_list'].append(case_info)
    return dataset_info


def download_metainfos(case_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    metainfos = {}
    num_cases = len(case_ids)
    total_batches = (num_cases // 100) + (1 if num_cases % 100 != 0 else 0)
    for row in tqdm(range(total_batches), desc='Obtaining case metainfos', unit='batch'):
        start_row_index = row * 100
        end_row_index = start_row_index + 100
        batch_case_ids = case_ids[start_row_index:end_row_index]
        results = get_metainfo_by_case_id(batch_case_ids)
        for res in results:
            metainfos[res['case_id']] = res
    return metainfos
    

def main(max_tissue_size: int, output_path: Path, num_workers: int = 5):
    tissue_data = pd.concat([pd.read_csv(path, sep='\t', header=0, index_col=0) for path in tissue_manifests], axis=0)
    rna_seq_data = pd.concat([pd.read_csv(path, sep='\t', header=0, index_col=0) for path in rna_seq_manifests], axis=0)
    
    tissue_data = tissue_data[tissue_data['size'] < max_tissue_size]
    tissue_cases_df = get_cases_info_from_file_ids(tissue_data)
    rna_seq_cases_df = get_cases_info_from_file_ids(rna_seq_data)
    
    case_ids = tissue_cases_df['case_id'].unique().tolist()  
    rna_seq_cases_df = rna_seq_cases_df[rna_seq_cases_df['case_id'].isin(case_ids)]
    
    dataset_info = build_dataset_info(tissue_cases_df, rna_seq_cases_df)
    dataset_info_path = output_path / 'dataset_info.pkl'
    with open(dataset_info_path, 'wb') as f:
        pickle.dump(dataset_info, f)
    logging.info(f'Dataset info saved to {dataset_info_path} ({dataset_info["metainfo"]["num_cases"]} cases)')
    
    tissue_file_names = sum([f['tissue_files'] for f in dataset_info['data_list']], [])
    rna_seq_file_names = sum([f['rna_seq_files'] for f in dataset_info['data_list']], [])
    
    portion_submitter_ids = clean_rna_seq(rna_seq_file_names, output_path / 'rna_seq', dataset_info_path)
    clean_tissue_slides(tissue_file_names, output_path / 'tissue', dataset_info_path, portion_submitter_ids)
    logging.info(f'Cleaned rna-seq and tissue slides.')
    
    with open(dataset_info_path, 'rb') as f:
        dataset_info = pickle.load(f)
    dataset_info['metainfo']['num_cases'] = len(dataset_info['data_list'])
    
    with open(dataset_info_path, 'wb') as f:
        pickle.dump(dataset_info, f)
    logging.info(f'Updated dataset info saved to {dataset_info_path} ({dataset_info["metainfo"]["num_cases"]} cases)')
    
    case_ids = [d['case_id'] for d in dataset_info['data_list']]
    with open(output_path / 'case_ids.txt', 'w') as f:
        for case_id in case_ids:
            f.write(f'{case_id}\n')
    logging.info(f'Case IDs saved to {output_path / "case_ids.txt"}')
            
    metainfos = download_metainfos(case_ids)
    metainfos_path = output_path / 'metainfos.pkl'
    with open(metainfos_path, 'wb') as f:
        pickle.dump(metainfos, f)
    logging.info(f'Metainfos saved to {metainfos_path}')
    
    tissue_files = set([f for d in dataset_info['data_list'] for f in d['tissue_files']])
    rna_seq_files = set([f for d in dataset_info['data_list'] for f in d['rna_seq_files']])
    
    tissue_cases_df = tissue_cases_df[tissue_cases_df['file_name'].isin(tissue_files)]
    rna_seq_cases_df = rna_seq_cases_df[rna_seq_cases_df['file_name'].isin(rna_seq_files)]
    
    tissue_cases_df.to_parquet(str(output_path / 'tissue_cases.parquet'), index=False)
    rna_seq_cases_df.to_parquet(str(output_path / 'rna_seq_cases.parquet'), index=False)
    
    image_encoder = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    image_encoder = image_encoder.cuda()
    image_encoder.eval()
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    download_and_preprocess_tissues(image_encoder, transform, tissue_cases_df, output_path / 'patch_embeddings_uni', 2)
    logging.info(f'Downloaded {tissue_cases_df.shape[0]} tissue files and {rna_seq_cases_df.shape[0]} RNA-seq files')
    logging.info(f'Downloaded {tissue_cases_df.shape[0] + rna_seq_cases_df.shape[0]} files in total')
    logging.info(f'All files downloaded to {(output_path)}')
    
    del image_encoder, transform
    gc.collect()
    torch.cuda.empty_cache()
    
    download_files_parallel(rna_seq_cases_df, output_path / 'rna_seq', num_workers)
    
    # filter files that are not in dataset_info
    tissue_files = set([f for d in dataset_info['data_list'] for f in d['tissue_files']])
    rna_seq_files = set([f for d in dataset_info['data_list'] for f in d['rna_seq_files']])
    
    for f in os.listdir(output_path / 'tissue'):
        if f not in tissue_files:
            os.remove(output_path / 'tissue' / f)
            print(f"🗑️ Rimosso: {f}")
            
    for f in os.listdir(output_path / 'rna_seq'):
        if f not in rna_seq_files:
            os.remove(output_path / 'rna_seq' / f)
            print(f"🗑️ Rimosso: {f}")
    
    load_rna_seq_dataframe(str(output_path / 'rna_seq'), dataset_info_path, str(output_path / 'rna_seq.parquet'))
    logging.info(f'RNA-seq dataframe saved to {output_path / "rna_seq.parquet"}')
    save_case_id_to_slide_file_dataframe(dataset_info_path, str(output_path / 'slides_info.parquet'))
    logging.info(f'Slides info dataframe saved to {output_path / "slides_info.parquet"}')
    logging.info(f'All files downloaded and cleaned. Dataset ready for use!')
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Script to download files from GDC")
    parser.add_argument("-max_tissue_size", default=70000000, type=int, required=False,
                        help="Maximum size of tissue files to download (in bytes)")
    parser.add_argument("-output_path", default=None, type=str, required=True,
                        help="Output path for downloaded files")
    parser.add_argument("-num_workers", default=5, type=int, required=False,
                        help="Number of workers for downloading files")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
    logging.basicConfig(filename=str(output_path / 'log.txt'), filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(
        max_tissue_size=args.max_tissue_size,
        output_path=output_path,
        num_workers=args.num_workers
    )
    
    
    
    
    
