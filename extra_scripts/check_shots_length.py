import os
from glob import glob
import pandas as pd
import pickle

shots = glob('dataset/5_CV/split_full/train/*.pkl')

min_len = 100000

for s in shots:
    with open(s, 'rb') as f:
        plasma_shot_df = pickle.load(f)

    if plasma_shot_df.shape[0] < min_len:
        min_len = plasma_shot_df.shape[0]

print(min_len)
