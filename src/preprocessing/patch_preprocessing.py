import pickle
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import openslide
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


def otsu_mask_skimage(slide, level=6):
    thumbnail = np.array(slide.read_region((0, 0), 0, slide.level_dimensions[level]))[..., :3]
    
    gray = rgb2gray(thumbnail)
    thresh = threshold_otsu(gray)
    mask = gray < thresh

    return mask.astype(np.uint8)

def extract_tiles(slide, mask, tile_size=512, background_thresh=0.8, mask_level=6):
    mask_dims = slide.level_dimensions[mask_level]
    full_dims = slide.level_dimensions[0]
    scale_x = full_dims[0] / mask_dims[0]
    scale_y = full_dims[1] / mask_dims[1]

    num_x = (full_dims[0] // tile_size) + 1
    num_y = (full_dims[1] // tile_size) + 1

    valid_tiles = []
    for y in range(num_y):
        for x in range(num_x):
            x0 = x * tile_size
            y0 = y * tile_size

            mx = int(x0 / scale_x)
            my = int(y0 / scale_y)
            mw = int(tile_size / scale_x)
            mh = int(tile_size / scale_y)

            tile_mask = mask[my:my+mh, mx:mx+mw]
            if tile_mask.shape[0] == 0 or tile_mask.shape[1] == 0:
                continue

            tissue_fraction = np.count_nonzero(tile_mask) / tile_mask.size
            if tissue_fraction < (1 - background_thresh):
                continue
            
            size_x = min(tile_size, full_dims[0] - x0)
            size_y = min(tile_size, full_dims[1] - y0)
            valid_tiles.append((x0, y0, size_x, size_y))
    return valid_tiles


def main(dataset_path: Path):
    
    with open(dataset_path / 'dataset_info.pkl', 'rb') as f:
        infos = pickle.load(f)['data_list']
        
    valid_tiles = {}

    for info in tqdm(infos):
        case_id = info['case_id']
        case_slide_file_name = info['tissue_files'][0]
        slide_path = dataset_path / 'tissue' / case_slide_file_name
        
        slide = openslide.OpenSlide(slide_path)
        mask = otsu_mask_skimage(slide, 0)
        
        tiles_1024 = extract_tiles(slide, mask, tile_size=1024, background_thresh=0.8, mask_level=0)
        tiles_512 = extract_tiles(slide, mask, tile_size=512, background_thresh=0.8, mask_level=0)
        tiles_256 = extract_tiles(slide, mask, tile_size=256, background_thresh=0.8, mask_level=0)
        tiles_128 = extract_tiles(slide, mask, tile_size=128, background_thresh=0.8, mask_level=0)
        valid_tiles[case_id] = {
            1024: tiles_1024,
            512: tiles_512,
            256: tiles_256,
            128: tiles_128,
        }
    
    with open(dataset_path / 'valid_patches.json', 'w') as f:
        json.dump(valid_tiles, f, indent=4)
        
        
def parse_args():
    parser = argparse.ArgumentParser(description="Extract valid patches from slides.")
    parser.add_argument("--dataset_path", type=str, default="/CompanyDatasets/carlos00/dataset_project", help="Path to the dataset")
    return parser.parse_args()
        
if __name__ == "__main__":
    args = parse_args()
    main(Path(args.dataset_path))