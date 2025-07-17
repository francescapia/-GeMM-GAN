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

from utils import (
    map_gdc_file, 
    get_metainfo_by_case_id, 
    download_file_by_id, 
    clean_rna_seq, 
    clean_tissue_slides,
    load_rna_seq_dataframe,
    save_case_id_to_slide_file_dataframe
)


tissue_manifests = [
    './LLM_Project_2025/data/gdc_manifest.2025-04-18.223014_tissue_normal.txt',
    './LLM_Project_2025/data/gdc_manifest.2025-04-18.223200_tissue_tumor_unknown_specimen.txt',
    './LLM_Project_2025/data/gdc_manifest.2025-04-18.223419_tissue_tumor_solid_tissue_specimen_ffpe_oct_preservation.txt',
    './LLM_Project_2025/data/gdc_manifest.2025-04-18.223419_tissue_tumor_solid_tissue_specimen_unknown_preservation.txt'
]

rna_seq_manifests = [
    './LLM_Project_2025/data/gdc_manifest.2025-04-18.221417_rna_seq_part1.txt',
    './LLM_Project_2025/data/gdc_manifest.2025-04-18.221646_rna_seq_part2.txt',
]


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
    
    portion_submitter_ids = clean_rna_seq(output_path / 'rna_seq', dataset_info_path)
    clean_tissue_slides(output_path / 'tissue', dataset_info_path, portion_submitter_ids)
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
    
    download_files_parallel(tissue_cases_df, output_path / 'tissue', num_workers)
    download_files_parallel(rna_seq_cases_df, output_path / 'rna_seq', num_workers)
    logging.info(f'Downloaded {tissue_cases_df.shape[0]} tissue files and {rna_seq_cases_df.shape[0]} RNA-seq files')
    logging.info(f'Downloaded {tissue_cases_df.shape[0] + rna_seq_cases_df.shape[0]} files in total')
    logging.info(f'All files downloaded to {(output_path)}')
    
    # filter files that are not in dataset_info
    # tissue_files = set([f for d in dataset_info['data_list'] for f in d['tissue_files']])
    # rna_seq_files = set([f for d in dataset_info['data_list'] for f in d['rna_seq_files']])
    
    # for f in os.listdir(output_path / 'tissue'):
    #     if f not in tissue_files:
    #         os.remove(output_path / 'tissue' / f)
    #         print(f"🗑️ Rimosso: {f}")
            
    # for f in os.listdir(output_path / 'rna_seq'):
    #     if f not in rna_seq_files:
    #         os.remove(output_path / 'rna_seq' / f)
    #         print(f"🗑️ Rimosso: {f}")
    
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
    
    
    
    
    
