import pickle
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import json
import argparse
from pathlib import Path
import logging


system_prompt = """
You are a assisting an AI engineer to generate synthetic gene expression data.
You are given a metadata dictionary that contains information about a patient.
The metadata dictionary contains information about a patient, including their demographics, 
diagnoses history, and other relevant information.

Your task is to generate a very small description (around 200 words in a single paragraph) of the patient based 
on the metadata provided, which contains many irrelevant fields. 
The description should be concise and relevant to gene expression data, i.e. it should contain the type of desease 
and the site in which is located, together with other demographic information about the patient and some contextual
information about the experiments and treatments, if available. Please be impersonal and do not mention the specific 
patient but be generic. Return a single paragraph, do not use bullet points or enumerations.
The description should be in English and should not contain any special characters or formatting.
Do not include any questions in the output, just an objective description of the patient. If more diagnoses or treatments are available,
please include all of them in the description, without trying to infer anything about the correct one.
Insert the string "[/INST]" at the beginning of the description to differentiate it from the metadata.
"""


def main(model_name: str, dataset_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cuda:0"
    )
    
    with open(dataset_path / 'metainfos.pkl', 'rb') as f:
        metadata = pickle.load(f)

    with open(dataset_path / 'case_ids.txt', 'r') as f:
        case_ids = [line.strip() for line in f.readlines()]
        
    descriptions = {}

    # case_ids = list(metadata.keys())
    for case_id in tqdm(case_ids):
        case_metadata = metadata[case_id]
        
        if case_metadata is None:
            descriptions[case_id] = ''
            logging.warning(f"Case {case_id} has no metadata.")
            continue
        
        del case_metadata['samples']
        del case_metadata['case_id']
        del case_metadata['submitter_id']
        del case_metadata['project']
        if 'demographic' in case_metadata and case_metadata['demographic'] is not None:
            if 'demographic_id' in case_metadata['demographic']:
                del case_metadata['demographic']['demographic_id']
            if 'updated_datetime' in case_metadata['demographic']:
                del case_metadata['demographic']['updated_datetime']
            if 'created_datetime' in case_metadata['demographic']:
                del case_metadata['demographic']['submitter_id']
            if 'days_to_birth' in case_metadata['demographic']:
                del case_metadata['demographic']['days_to_birth']
        
        metadata_str = []
        for key, value in case_metadata.items():
            if isinstance(value, dict):
                metadata_str.append(f"{key}: {', '.join([f'{k}: {v}' for k, v in value.items() if v is not None])}")
            elif isinstance(value, list):
                metadata_str.append(f"{key}: {', '.join([str(v) for v in value])}")
            else:
                metadata_str.append(f"{key}: {value}")
        metadata_str = "\n".join(metadata_str)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": metadata_str}
        ]
        
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id
            )

        description = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if '[/INST]' not in description or description.count('[/INST]') == 1:
            # if count is 1, it means that INST is only in the repeated part of the message
            description = description.split('\n')[-1].strip()
        else:
            description = description.split('[/INST]')[-1].strip()
        
        descriptions[case_id] = description
    
    with open(dataset_path / 'descriptions.json', 'w') as f:
        json.dump(descriptions, f, indent=4)
        
        
def parse_args():
    parser = argparse.ArgumentParser(description="Generate text descriptions from case metadata.")
    parser.add_argument("--model_name", type=str, default="ContactDoctor/Bio-Medical-Llama-3-8B", help="Model name")
    parser.add_argument("--dataset_path", type=str, default="/CompanyDatasets/carlos00/all_tcga_100mb", help="Path to the dataset")
    return parser.parse_args()
        
if __name__ == "__main__":
    args = parse_args()
    main(args.model_name, Path(args.dataset_path))