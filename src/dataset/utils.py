import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict
import logging
import os
import shutil
import pickle
from tqdm import tqdm


def download_file_by_id(file_id: str, output_path: Path, num_retries: int = 2) -> bool:
    url = f"https://api.gdc.cancer.gov/data/{file_id}"

    for i in range(num_retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except Exception as e:
            logging.info(f"[{i}] Failed to download {file_id}: {e}")
            continue
    logging.info(f"❌ Failed to download {file_id} after {num_retries} attempts.")
    return False


def map_gdc_file(file_id: Union[List[str], str]) -> Union[pd.DataFrame, None]:
    url = 'https://api.gdc.cancer.gov/files'
    
    payload = {
        "filters": {
            "op": "in",
            "content": {
                "field": "file_id",
                "value": file_id if isinstance(file_id, list) else [file_id]
            }
        },
        "fields": "file_id,file_name,cases.submitter_id,cases.case_id,cases.project.project_id",
        "format": "JSON",
        "size": 100
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        data = response.json()
        hits = data.get("data", {}).get("hits", [])
        
        if not hits:
            print("No results found for file_id:", file_id)
            return None
        
        results = []
        for info in response.json()['data']['hits']:
            results.append(
                {
                    'file_id': info['file_id'],
                    'file_name': info['file_name'],
                    'submitter_id': info['cases'][0]['submitter_id'],
                    'case_id': info['cases'][0]['case_id'],
                    'project_id': info['cases'][0]['project']['project_id']
                }
            )
        results = pd.DataFrame(results)
        return results

    logging.info(f"Error: {response.status_code}")
    logging.info(response.text)
    return None


def get_metainfo_by_case_id(case_ids) -> List[Dict]:
    endpoint = "https://api.gdc.cancer.gov/cases"

    filters = {
        "op": "in",
        "content": {
            "field": "case_id",
            "value": case_ids
        }
    }

    fields = [
        "case_id",
        "submitter_id",
        "project.project_id"
        "index_date",
        "state",
        # clinical data
        "follow_ups.*",
        "diagnoses.*",
        "demographic.*",
        # biospecimen data
        "samples.*",
    ]

    params = {
        "filters": filters,
        "expand": ",".join(fields),
        "format": "JSON",
        "size": len(case_ids)
    }

    response = requests.post(endpoint, json=params)
    response.raise_for_status()
    
    fields_to_save = ['case_id', 'submitter_id', 'project', 'disease_type', 'primary_site', 'demographic', 'diagnoses', 'samples']
    data = response.json().get("data", {}).get("hits", [])
    if not data:
        print("⚠️ Nessun risultato trovato.")
        return []
    
    hits = []
    for hit in data:
        hit_data = {}
        for field in fields_to_save:
            if field in hit:
                hit_data[field] = hit[field]
            else:
                hit_data[field] = None
        hits.append(hit_data)
    return hits


def clean_rna_seq(data_dir: str, dataset_path: str) -> None:
    
    #Provide the RNA-seq directory and the path to the dataset_info.pkl file
    
    url = 'https://api.gdc.cancer.gov/files'
    file_names = [f for f in os.listdir(data_dir) if f.endswith(".tsv")]
    
    fields = [
        "file_id", 
        "file_name", 
        "cases.submitter_id", 
        "cases.case_id", 
        "cases.project.project_id",
        "cases.samples.tissue_type", 
        "cases.samples.tumor_descriptor",
        # "cases.samples.portions.slides.section_location", 
        # "cases.samples.portions.slides.slide_id",
        "cases.samples.portions.submitter_id",
        "cases.samples.portions.analytes.submitter_id",
        "cases.samples.portions.analytes.aliquots.submitter_id",
        "created_datetime"
    ]
    payload = {
        "filters": {
            "op": "in",
            "content": {
                "field": "file_name",
                "value": None
            }
        },
        "fields": ','.join(fields),
        "format": "JSON",
        "size": 100
    }
    
    # added batches since api has a limit of 100 files per request
    num_files = len(file_names)
    total_batches = (num_files // 100) + (1 if num_files % 100 != 0 else 0)
    total_hits = []
    for row in tqdm(range(total_batches), desc='Obtaining files metainfos', unit='batch'):
        start_row_index = row * 100
        end_row_index = start_row_index + 100
        batch_files = file_names[start_row_index:end_row_index]
        payload["filters"]["content"]["value"] = batch_files

        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"Errore nella richiesta: {response.status_code}\n{response.text}")
            return

        hits = response.json().get("data", {}).get("hits", [])
        if not hits:
            print("⚠️ Nessun risultato trovato.")
            return
        total_hits.extend(hits)

    df = pd.DataFrame([{
        'file_id': h['file_id'],
        'file_name': h['file_name'],
        'submitter_id': h['cases'][0]['submitter_id'],
        'case_id': h['cases'][0]['case_id'],
        'project_id': h['cases'][0]['project']['project_id'],
        'tissue_type': h['cases'][0]['samples'][0]['tissue_type'],
        'tumor_descriptor': h['cases'][0]['samples'][0]['tumor_descriptor'],
        'portion_submitter_id': h['cases'][0]['samples'][0]['portions'][0]['submitter_id'],
        'analyte_submitter_id': h['cases'][0]['samples'][0]['portions'][0]['analytes'][0]['submitter_id'],
        'aliquot_submitter_id': h['cases'][0]['samples'][0]['portions'][0]['analytes'][0]['aliquots'][0]['submitter_id'],
        'created_datetime': h['created_datetime']
    } for h in total_hits])
    df['created_datetime'] = pd.to_datetime(df['created_datetime'])
    
    all_files = set(df['file_name'])
    
    df = df.sort_values(by=['case_id', 'created_datetime'], ascending=[True, False])
    df = df.drop_duplicates(subset='case_id', keep='first')
    
    valid_mask = (df['tissue_type'] != 'Normal') & (df['tumor_descriptor'] == 'Primary')
    df = df[valid_mask]
    
    files_to_keep = set(df['file_name'])
    files_to_remove = all_files - files_to_keep
    
    for file in files_to_remove:
        fpath = os.path.join(data_dir, file)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"🗑️ Rimosso: {file}")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    for entry in dataset['data_list']:
        entry['rna_seq_files'] = [f for f in entry['rna_seq_files'] if f in files_to_keep]

    dataset['data_list'] = [e for e in dataset['data_list'] if e['tissue_files'] and e['rna_seq_files']]

    backup_path = str(dataset_path) + ".bak"
    shutil.copy(dataset_path, backup_path)
    print(f"🔒 Backup salvato in: {backup_path}")

    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    return df['portion_submitter_id'].unique().tolist()
        

def clean_tissue_slides(data_dir: str, dataset_path: str, valid_portion_submitter_ids: List[str]) -> None:
    
    #Provide the Tissue directory and the path to the dataset_info.pkl file
    
    url = 'https://api.gdc.cancer.gov/files'
    file_names = [f for f in os.listdir(data_dir) if f.endswith(".svs")]
    
    fields = [
        "file_id", 
        "file_name", 
        "cases.submitter_id", 
        "cases.case_id", 
        "cases.project.project_id",
        "cases.samples.tissue_type", 
        "cases.samples.tumor_descriptor",
        "cases.samples.portions.slides.section_location", 
        "cases.samples.portions.slides.slide_id",
        "cases.samples.portions.submitter_id",
        # "cases.samples.portions.analytes.submitter_id",
        # "cases.samples.portions.analytes.aliquots.submitter_id",
        "created_datetime"
    ]
    payload = {
        "filters": {
            "op": "in",
            "content": {
                "field": "file_name",
                "value": None
            }
        },
        "fields": ','.join(fields),
        "format": "JSON",
        "size": 100
    }
    
    # added batches since api has a limit of 100 files per request
    num_files = len(file_names)
    total_batches = (num_files // 100) + (1 if num_files % 100 != 0 else 0)
    total_hits = []
    for row in tqdm(range(total_batches), desc='Obtaining files metainfos', unit='batch'):
        start_row_index = row * 100
        end_row_index = start_row_index + 100
        batch_files = file_names[start_row_index:end_row_index]
        payload["filters"]["content"]["value"] = batch_files

        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"Errore nella richiesta: {response.status_code}\n{response.text}")
            return

        hits = response.json().get("data", {}).get("hits", [])
        if not hits:
            print("⚠️ Nessun risultato trovato.")
            return
        total_hits.extend(hits)
    
    df = []
    for h in total_hits:
        slide_id = h['file_name'].split('.')[1].lower()
        try:
            section_location = [x['section_location'] for x in h['cases'][0]['samples'][0]['portions'][0]['slides']
                                if x['slide_id'] == slide_id][0]
        except IndexError:
            print(h['file_name'], slide_id, h['cases'][0]['samples'][0]['portions'][0]['slides'])
            section_location = 'UNKNOWN'
            
        df.append({
            'file_id': h['file_id'],
            'file_name': h['file_name'],
            'submitter_id': h['cases'][0]['submitter_id'],
            'case_id': h['cases'][0]['case_id'],
            'project_id': h['cases'][0]['project']['project_id'],
            'tissue_type': h['cases'][0]['samples'][0]['tissue_type'],
            'tumor_descriptor': h['cases'][0]['samples'][0]['tumor_descriptor'],
            'section_location': section_location,
            'portion_submitter_id': h['cases'][0]['samples'][0]['portions'][0]['submitter_id'],
            'created_datetime': h['created_datetime']
        })
    df = pd.DataFrame(df)
    df['created_datetime'] = pd.to_datetime(df['created_datetime'])
    
    print(df['section_location'].unique().tolist())
    print(f'Bottom slides: {(df['section_location'] == 'BOTTOM').sum()}')
    print(f'Top slides: {(df['section_location'] == 'TOP').sum()}')
    
    all_files = set(df['file_name'])
    df = df[df['portion_submitter_id'].isin(valid_portion_submitter_ids)]
    
    # by sorting by section location before created datetime in ascending order, we select the most recent bottom slide 
    # if present, otherwise we selct the most recent top slide
    # if slide_id do not allow to distinguish between bottom and top, we select the most recent one (unknown section location)
    df = df.sort_values(by=['case_id', 'section_location', 'created_datetime'], ascending=[True, True, False])
    df = df.drop_duplicates(subset='case_id', keep='first')
      
    valid_mask = (df['tissue_type'] != 'Normal') & (df['tumor_descriptor'] == 'Primary')
    df = df[valid_mask]
    
    files_to_keep = set(df['file_name'])
    files_to_remove = all_files - files_to_keep
    
    for file in files_to_remove:
        fpath = os.path.join(data_dir, file)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"🗑️ Rimosso: {file}")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    for entry in dataset['data_list']:
        entry['tissue_files'] = [f for f in entry['tissue_files'] if f in files_to_keep]

    dataset['data_list'] = [e for e in dataset['data_list'] if e['tissue_files'] and e['rna_seq_files']]

    backup_path = str(dataset_path) + "_.bak"
    shutil.copy(dataset_path, backup_path)
    print(f"🔒 Backup salvato in: {backup_path}")

    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
        
        
def load_dataset_info(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_rna_seq_dataframe(base_path, dataset_info_path, output_path, log_transform=True):
    """
    Loads RNA-seq data from individual files, filters for protein-coding genes,
    and returns a combined DataFrame with samples as rows and genes as columns.

    Parameters:
        base_path (str): Base directory containing RNA-seq files.
        dataset_info_path (str): Path to the dataset_info.pkl file.
        log_transform (bool): Whether to apply log2(x + 1) transformation.

    Returns:
        pd.DataFrame: Final RNA-seq DataFrame (samples x genes).
    """
    all_samples = []
    dataset = load_dataset_info(dataset_info_path)

    for sample in tqdm(dataset['data_list'], desc='Loading RNA-seq data', unit='sample'):
        case_id = sample['case_id']
        rna_filename = sample['rna_seq_files'][0]  # Assuming one file per sample

        file_path = os.path.join(base_path, rna_filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path, sep="\t", header=1)
        df = df.iloc[4:, :]
        df = df[df['gene_type'] == 'protein_coding']
        df = df[['gene_id', 'tpm_unstranded']]
        df = df.set_index('gene_id')
        df = df.rename(columns={'tpm_unstranded': case_id})

        all_samples.append(df)

    final_df = pd.concat(all_samples, axis=1).T  # samples x genes

    if log_transform:
        final_df = np.log2(final_df + 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path)
    print(f"RNA-seq data saved to {output_path}")
    print(f"RNA-seq data shape: {final_df.shape}")
    return final_df


def save_case_id_to_slide_file_dataframe(dataset_info_path, output_path):
    dataset = load_dataset_info(dataset_info_path)

    case_ids = []
    filenames = []
    for sample in dataset['data_list']:
        case_ids.append(sample['case_id'])
        filenames.append(sample['tissue_files'][0])  # Assuming one file per sample

    df = pd.DataFrame({'case_id': case_ids, 'file_name': filenames})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"Mapping case -> tissue slide saved to {output_path}")
    return df
