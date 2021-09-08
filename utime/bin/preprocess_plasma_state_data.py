import numpy as np
import pandas as pd
from utime.bin.helper_funcs import load_shot_and_equalize_times, normalize_signals_mean 
import pickle
import time
from collections import defaultdict
import argparse
import os
from glob import glob
import scipy.io

def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = argparse.ArgumentParser(description="Preprocess data for dataset preparation")
    parser.add_argument("--data_dir", type=str,
                        help="Path to output data directory")
    parser.add_argument("--machine", type=str,
                        help="Specify machine")
    parser.add_argument("--test", type=bool,default=False,
                        help="Git unit test boolean")
    return parser

def first_prepro_cycle(shots_and_labelers, shots_and_labelers_dic, data_dir, machine):
    
    intc_times = {}
    shot_dfs = {}
    
    for s in shots_and_labelers:
        
        labelers = shots_and_labelers_dic[s.split('-')[0]]
        
        # Load .csv data -- apply IP cut -- remove_no_state -- Normalize IP -- remove_disruption_points 
        fshot, intc_times[s.split('-')[0]] = load_shot_and_equalize_times(data_dir, s.split('-')[0], labelers, machine)
        
        shot_dfs[str(s)] = fshot.copy()

    return shot_dfs, intc_times

def second_prepro_cycle(shots_and_labelers, shot_dfs, itsc_times, machine):
    
    # Normalize input signals (minmax by default is applied)
    for shot in shots_and_labelers:
        shot_no = shot[:5]
        labeler_intersect_times = itsc_times[shot_no]
        fshot = shot_dfs[str(shot)].copy()
        fshot = fshot[fshot['time'].round(5).isin(labeler_intersect_times)]
        fshot = normalize_signals_mean(fshot, machine) #NORMALIZATION CAN ONLY HAPPEN AFTER SHOT FROM BOTH LABELERS HAS BEEN ASSERTED TO BE THE SAME!
        shot_dfs[str(shot)] = fshot

    return shot_dfs

def PreprocessingOffline(data_dir_in, data_dir_out, shot_ids, labelers, machine, unit_test):
    ''' Preprocess data offline. This is intended to use it just one
    and hence avoid performing same preprocessing each time for running
    an experiment.'''
    
    shots_and_labelers = []
    shots_and_labelers_dic = defaultdict(list)

    for s in shot_ids:
        for l in labelers:
            # Skip preprocessing if it was already done
            if os.path.exists(os.path.join(data_dir_out,'shot_{}.pkl'.format(str(s) + '-' + str(l)))):
                continue
            if os.path.exists(data_dir_in + l + '/' + machine + '_'  + str(s).split('-')[0] + '_' + l + '_labeled.csv'):
                shots_and_labelers += (str(s) + '-' + str(l),)
                shots_and_labelers_dic[str(s)].append(l)
    
    if not shots_and_labelers:
        return

    shot_dfs, itsc_times = first_prepro_cycle(shots_and_labelers, shots_and_labelers_dic, data_dir_in, machine)
    
    if not shot_dfs:
        return

    shot_dfs = second_prepro_cycle(shots_and_labelers, shot_dfs, itsc_times, machine)
    if unit_test:
        dirpath = os.getcwd()
        pd = np.asarray(shot_dfs['87870-marceca']['PD'].values)
        if 'algorithms' in dirpath:
            scipy.io.savemat(os.path.join(dirpath, './unit_tests/prep.mat'), {'PD': pd})
        else:
            scipy.io.savemat(os.path.join(dirpath, './algorithms/GMUTime/UTime-PlasmaStates/unit_tests/prep.mat'), {'PD': pd})
    else:
        # Save pre-process data
        for i, sdf in enumerate(shot_dfs.values()):
            with open(os.path.join(data_dir_out,'shot_{}.pkl'.format(list(shot_dfs.keys())[i])), 'wb') as f:
                pickle.dump(sdf, f)
    
def run(args):
    '''
    This function reads the .csv validated shots, pre-processed them: apply IP cut, remove_no_state, Normalize IP, remove_disruption_points and normalize signals (minmax)
    Inputs: Clusters_*.mat which contains an array `clusters` and a cell `shotlist`. This mapping
    assigns a given shot number to a certain cluster.
    Output: data preprocessed and stored in shot_*.pkl format
    '''
    
    unit_test = args.test
    machine = args.machine
 
    data_dir_in = os.path.abspath(args.data_dir)
    data_dir_in = os.path.join(data_dir_in, 'Validated/')

    data_dir_out = os.path.abspath(args.data_dir)
    data_dir = os.path.abspath(args.data_dir)
    start = time.time()
    labelers = ['marceca']

    # Get Clusters Hierarchy structure and preprocess shots
    # Load a matlab .mat file as np arrays
    if not unit_test:
        from scipy.io import loadmat
        if machine == 'TCV':
            DWT_out = loadmat(os.path.join(data_dir, 'Clusters_DWT_26052020.mat'))
        elif machine == 'JET':
            DWT_out = loadmat(os.path.join(data_dir, 'Clusters_ES_JET_v1.mat'))
        # Get shotlist and clusters obtained from DTW tool
        shotlist = DWT_out['shotlist']
        if machine == 'TCV':
            clusters = DWT_out['myclusters']
        elif machine == 'JET':
            clusters = DWT_out['clusters']
        
        # Flat arrays
        shotlist = [item[0][0].flat[0] for item in shotlist[0]]
        clusters = [item.flat[0] for item in clusters[0]]
        all_shots = [int(f_.split('/')[-1].split('_')[1]) for lab in labelers for f_ in glob(os.path.join(data_dir_in, lab + '/' + machine + '*'))]
        #print('all_shots: ', all_shots)

        # Check all shots in dir are in the Cluster structure
        for s in all_shots:
            if s not in shotlist:
                print('shot {} not in cluster structure. Assigning it to a new cluster'.format(s))
                residual_cluster = np.max(clusters) + 1
                clusters.append(residual_cluster)
                shotlist.append(s)

    else:
        shotlist = [87870]
        clusters = [1]

    n_clusters = len(set(clusters))
    
    for i in np.arange(1, n_clusters+1):
        
        print('Start preprocessing of cluster %i' % i)

        # Get shots associated with cluster i
        shots = np.array(shotlist)[np.where(np.array(clusters) == i)[0]]
        data_dir_tmp = os.path.join(data_dir_out, 'cluster_{}'.format(str(i)))
        
        if not os.path.exists(data_dir_tmp):
            print("Creating directory at %s" % data_dir_tmp)
            os.makedirs(data_dir_tmp)

        PreprocessingOffline(data_dir_in, data_dir_tmp, shots, labelers, machine, unit_test)
    
    end = time.time()
    print('elapsed time for data preprocessing: ', end - start)


def entry_func(args=None):
    parser = get_argparser()
    run(parser.parse_args(args))

if __name__ == "__main__":
    entry_func()
