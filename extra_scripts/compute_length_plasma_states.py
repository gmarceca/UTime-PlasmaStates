import os
from glob import glob
import pandas as pd
import pickle
from itertools import groupby
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

#shots = glob('../dataset/test/*.pkl')
shots = glob('../dataset/5_CV/split_full/train/*.pkl')
#shots = glob('../dataset/test/shot_62744-ffelici.pkl')

length_states = defaultdict(list)

for s in shots:
    with open(s, 'rb') as f:
        plasma_shot_df = pickle.load(f)
    
    #print(plasma_shot_df.shape[0])
    count_dups = [{_:sum(1 for _ in group)} for _, group in groupby(plasma_shot_df['LHD_label'].values)]
    #length_states.extend(count_dups)
    for count in count_dups:
        state = list(count.keys())[0]
        val = count[state]
        length_states[state].append(val)

bins_l = np.arange(0, 2000, 10)
bins_d = np.arange(0, 2000, 10)
bins_h = np.arange(0, 2000, 10)
#bins_l = np.arange(np.min(length_states[1]), np.max(length_states[1]), 10)
#bins_d = np.arange(np.min(length_states[2]), np.max(length_states[2]), 10)
#bins_h = np.arange(np.min(length_states[3]), np.max(length_states[3]), 10)

plt.hist(length_states[1], bins=bins_l, alpha=0.4, label='L')
plt.hist(length_states[2], bins=bins_l, alpha=0.4, label='D')
plt.hist(length_states[3], bins=bins_l, alpha=0.4, label='H')
plt.xlabel('Length plasma state')
plt.legend(loc='upper right')
plt.savefig('train_set_length_states.png')
#plt.close()
#plt.hist(length_states[2], bins=bins_d, alpha=0.4, label='D')
#plt.xlabel('Length state D')
#plt.savefig('test_set_length_state_D.png')
#plt.close()
#plt.hist(length_states[3], bins=bins_h, alpha=0.4, label='H')
#plt.xlabel('Length state H')
#plt.savefig('test_set_length_state_H.png')
#plt.close()
