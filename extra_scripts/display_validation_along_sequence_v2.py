import numpy as np
from glob import glob
import os
import pickle
import pandas as pd


# Inherited from https://gitlab.epfl.ch/spc/tcv/event-detection/blob/UTime-PlasmaStates/algorithms/FMLSTM/plot_routines.py #
def plot_all_signals_all_trans(times, TDAI, FIR, WP, pred, shot, slice_, points_per_window):
   
    from matplotlib import colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}
    import matplotlib
    matplotlib.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
    fig = plt.figure(figsize = (19, 5))
    leg = []
    p4 = fig.add_subplot(1,1,1)

    p4.plot(times,TDAI, label='TDAI')
    p4.plot(times,FIR, label='FIR')
    p4.plot(times,WP, label='WP')
    p4.grid()
    p4.set_ylabel('Signal values (norm.)')
    p4.set_xlabel('t(s)')
    
    p4.set_title('UTime predictions for shot {}'.format(str(shot)))

    colors_dic = {1:'yellow',2:'green',3:'blue'}
    
    for k in range(pred.shape[0]-1):
        #print(pred[k])
        if (points_per_window == 1):
            x_axis = times[k]
        else:
            x_axis = (times[(k+1)*points_per_window] + times[k*points_per_window])/2
        plt.vlines(x=x_axis, ymin=0, ymax=np.max(TDAI), linestyle='--', color = colors_dic[pred[k]], linewidth=2, alpha=0.5)

    
    import matplotlib.patches as mpatches
    
    output_dir = os.path.join('validation', 'shot_{}'.format(str(shot)))
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    #p4.legend(handles=[l_patch, d_patch, h_patch], loc=2, prop={'size': 16}, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_slice_{}_shot_{}.png'.format(slice_, shot)))
    #plt.show()
    
def downsample_labels (states, points_per_window):
    """
    """
    states = states.tolist()
    states_resampled = []
    for k in range(int(len(states)/points_per_window)):
        window = states[k*points_per_window : (k+1)*points_per_window]
        assert(len(window) == points_per_window)
        # determine the label of a window by getting the dominant class
        label = max(set(window), key = window.count)
        # L mode: label = 1, D mode: label = 2, H mode: label = 3
        # rest one so label is 0, 1 or 2
        #label -= 1
        states_resampled.append(label)
    return np.asarray(states_resampled)


#shot = 30268
#shot = 32911
#shot = 64770

shots_list = [87875]

for shot in shots_list:

    # Get predictions:
    #labelers = glob('TL_v4_new_dataset_project/eval/val_data/files/'+str(shot)+'-*/')
    #labeler = labelers[0]
    #lab = labeler.split('/')[-2]
    ## Get computed kappa value for this labeler
    #df = pd.read_csv('TL_v4_new_dataset_project/eval/val_data/evaluation_kappa.csv')
    ##print(df[lab])
    #df = df.rename(columns={"Unnamed: 0": "labelers"})
    #kappa_l = np.round(df.loc[df['labelers'] == lab]['cls 0'].values[0],2)
    #kappa_d = np.round(df.loc[df['labelers'] == lab]['cls 1'].values[0],2)
    #kappa_h = np.round(df.loc[df['labelers'] == lab]['cls 2'].values[0],2)
    
    #with np.load(labeler+'pred.npz') as data:
    #    pred = data['arr_0']

    #f = glob('../dataset_JET/1_CV/split_0/train/shot_' + str(shot) + '-marceca.pkl')[0]
    f = glob('../dataset_JET/test/shot_' + str(shot) + '-marceca.pkl')[0]
    with open(f, 'rb') as s:
        plasma_shot_df = pickle.load(s)
    pred = plasma_shot_df['LHD_label'].values 
    ## Get PD and times signals
    #shots_dir = 'new_dataset_plasma/test'
    #shot_file = os.path.join(shots_dir, 'shot_'+lab+'.pkl')
    #
    #with open(shot_file, 'rb') as f:
    #    shot_df = pickle.load(f)
    PD = plasma_shot_df.PD.values
    FIR = plasma_shot_df.FIR.values
    TDAI = plasma_shot_df.TDAI.values
    WP = plasma_shot_df.WP.values
    TP135 = plasma_shot_df.TP135.values
    times = plasma_shot_df.time.values
    
    # Plot slices (zoom)
    len_shot = TDAI.shape[0]
    len_seq = 10000
    points_per_window = 100

    pred = downsample_labels (pred, points_per_window)
    plot_slices = True
    if plot_slices:
        #for i in range(0, len_shot//len_seq):
        slices = np.arange(0, len_shot//len_seq + 1, 3)
        print('slices: ', slices)
        for s in range(0,len(slices)-1):
            
            for i in range(slices[s], slices[s+1]):
                times_ = times[i*len_seq:(i+1)*len_seq]
                TDAI_ = TDAI[i*len_seq:(i+1)*len_seq]
                FIR_ = FIR[i*len_seq:(i+1)*len_seq]
                WP_ = WP[i*len_seq:(i+1)*len_seq]
                pred_ = pred[i*len_seq//points_per_window:(i+1)*len_seq//points_per_window]
                if (s == len(slices)-2):
                    last_time_index = (i+1)*len_seq

                plot_all_signals_all_trans(times_, TDAI_, FIR_, WP_, pred_, shot, i, points_per_window)


        if (len_shot%len_seq):
            times_ = times[last_time_index:]
            TDAI_ = TDAI[last_time_index:]
            FIR_ = FIR[last_time_index:]
            WP_ = WP[last_time_index:]
            pred_ = pred[last_time_index//points_per_window:]

            plot_all_signals_all_trans(times_, TDAI_, FIR_, WP_, pred_, shot, 'end', points_per_window)
    else:
        plot_all_signals_all_trans(times, TDAI, FIR, WP, pred, shot, 'end', points_per_window)
