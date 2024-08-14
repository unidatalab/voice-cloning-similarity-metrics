import os
import argparse
import json
import logging
import pandas as pd
import seaborn as sns
from glob import glob
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from pyannote.audio import Model, Inference
from tqdm import tqdm
from easydict import EasyDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))

MIN_MOS = 1
MAX_MOS = 5

pyannote_model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.getenv('PYANNOTE_TOKEN'))

def mos_scale(x):
    return ((MAX_MOS - MIN_MOS) * (1 - x)) + MIN_MOS

def similarity_correction(x, max_x):

    corrected_value = MIN_MOS + (x - MIN_MOS) * (MAX_MOS - MIN_MOS) / (max_x - MIN_MOS)

    return corrected_value

def pyannote_dist(path1, path2):
    inference = Inference(pyannote_model, window="whole")
    emb1 = inference(path1)
    emb2 = inference(path2)

    distance = cdist([emb1], [emb2], metric="cosine")[0,0]

    return distance

def voice_similarity(originals_dir, cloned_dir, correction_type):

    if correction_type == 'interim':
        columns = ['orig_path', 'cloned_path', 'orig_sim', 'pyannote_sim', 'pyannote_sim_norm']
    else:
        columns = ['orig_path', 'cloned_path', 'orig_sim', 'pyannote_sim']

    df = pd.DataFrame(columns=columns)

    originals_dir_list = sorted(glob(f'{originals_dir}/*.wav'))
    cloned_dir_list = sorted(glob(f'{cloned_dir}/*.wav'))

    orig_cloned_comb_list = list(itertools.product(originals_dir_list, cloned_dir_list))
    orig_orig_comb_list = list(itertools.product(originals_dir_list, originals_dir_list))

    for orig_cloned_comb, orig_orig_comb in tqdm(zip(orig_cloned_comb_list, orig_orig_comb_list), total=len(orig_orig_comb_list)):

        orig_sim = pyannote_dist(orig_orig_comb[0], orig_orig_comb[1])
        pyannote_sim = pyannote_dist(orig_cloned_comb[0], orig_cloned_comb[1])

        pyannote_sim = mos_scale(pyannote_sim)
        orig_sim = mos_scale(orig_sim)

        if correction_type == 'interim':
            pyannote_sim_norm = similarity_correction(pyannote_sim, orig_sim)

            row = pd.DataFrame([[orig_cloned_comb[0], orig_cloned_comb[1], orig_sim, pyannote_sim, pyannote_sim_norm]], 
                            columns=columns)
        else:
            row = pd.DataFrame([[orig_cloned_comb[0], orig_cloned_comb[1], orig_sim, pyannote_sim]], 
                            columns=columns)    
        
        df = pd.concat([df, row], ignore_index=True)

    return df

def main(orig_dir, cloned_dir, correction_type, output_dir):
    
    df = voice_similarity(orig_dir, cloned_dir, correction_type)

    output_dir = Path(output_dir)

    df.to_csv(output_dir.joinpath('results.csv'), index=False)

    plot = sns.histplot(data=df)
    plt.xlim(1, 5)
    fig = plot.get_figure()
    fig.savefig(output_dir.joinpath('hist.png'))
    
    results = EasyDict()
    results.orig_orig_sim_mean = df['orig_sim'].mean().item()
    results.orig_clone_sim_mean = df['pyannote_sim'].mean().item()


    if correction_type == 'final':
        results.orig_clone_norm = similarity_correction(df['pyannote_sim'].mean(), df['orig_sim'].mean()).item()

    elif correction_type == 'interim':
        results.orig_clone_norm = df['pyannote_sim_norm'].mean().item()

    with open(output_dir.joinpath('stats.json'), 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(results)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--orig_dir', dest='orig_dir', type=str, help='Path to the directory that contains original speaker files', required=True)
    parser.add_argument('--cloned_dir', dest='cloned_dir', type=str, help='Path to the directory that contains cloned speaker files', required=True)
    parser.add_argument('--output_dir', dest='output_dir', type=str, help='Path to the directory where you want to have results', required=True)
    parser.add_argument('--correction_type', dest='correction_type', type=str, help='Selector of error correction_type', choices=['final', 'interim'])
    args = parser.parse_args()
    
    logger.info(args.orig_dir)
    main(args.orig_dir, args.cloned_dir, args.correction_type, args.output_dir)

