import numpy as np
from glob import glob
import os
import pickle
import pandas as pd


def plot_all_signals_all_trans_slices(times_list, PD_list, pred_list, shot, kappa_l, kappa_h, slice_, points_per_window):
   
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

    #fig, axs = plt.subplots(figsize=(19,5), nrows=3)
    fig, axs = plt.subplots(3,1,figsize=(19,12))
    #fig, axs = plt.subplots(3)

    leg = []

    for i in range(0,len(axs)):
        
        times = times_list[i]
        PD = PD_list[i]
        pred = pred_list[i]
        
        axs[i].plot(times,PD, label='PD')
        axs[i].grid()
        axs[i].set_ylabel('Signal values (norm.)')
        axs[i].set_xlabel('t(s)')

        colors_dic = {0:'yellow',1:'green',2:'blue'}
    
        for k in range(pred.shape[0]-1):
            #print(k)
            if (points_per_window == 1):
                x_axis = times[k]
            else:
                x_axis = (times[(k+1)*points_per_window] + times[k*points_per_window])/2
            axs[i].vlines(x=x_axis, ymin=0, ymax=np.max(PD), linestyle='--', color = colors_dic[pred[k]], linewidth=2, alpha=0.5)

        
    import matplotlib.patches as mpatches
    l_patch = mpatches.Patch(color='yellow', label='kappa score (L mode) = {}'.format(str(kappa_l)))
    h_patch = mpatches.Patch(color='blue', label='kappa score (H mode) = {}'.format(str(kappa_h)))

    fig.legend(handles=[l_patch, d_patch, h_patch], loc=3, prop={'size': 16}, ncol=1)
    
    output_dir = os.path.join('best_model_predictions', 'shot_{}'.format(str(shot)))
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fig.suptitle('UTime predictions for shot {}'.format(str(shot)))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_slice_{}_shot_{}.png'.format(slice_, shot)))
    plt.close()


# Inherited from https://gitlab.epfl.ch/spc/tcv/event-detection/blob/UTime-PlasmaStates/algorithms/FMLSTM/plot_routines.py #
def plot_all_signals_all_trans(times, PD, pred, shot, kappa_l, kappa_h, slice_, points_per_window):
   
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

    p4.plot(times,PD, label='PD')
    p4.grid()
    p4.set_ylabel('Signal values (norm.)')
    p4.set_xlabel('t(s)')
    
    p4.set_title('UTime predictions for shot {}'.format(str(shot)))

    colors_dic = {0:'yellow',1:'green',2:'blue'}
    
    for k in range(pred.shape[0]-1):
        #print(k)
        if (points_per_window == 1):
            x_axis = times[k]
        else:
            x_axis = (times[(k+1)*points_per_window] + times[k*points_per_window])/2
        plt.vlines(x=x_axis, ymin=0, ymax=np.max(PD), linestyle='--', color = colors_dic[pred[k]], linewidth=2, alpha=0.5)

    
    import matplotlib.patches as mpatches
    l_patch = mpatches.Patch(color='yellow', label='kappa score (L mode) = {}'.format(str(kappa_l)))
    h_patch = mpatches.Patch(color='blue', label='kappa score (H mode) = {}'.format(str(kappa_h)))
    
    output_dir = os.path.join('best_model_predictions', 'shot_{}'.format(str(shot)))
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    p4.legend(handles=[l_patch, d_patch, h_patch], loc=2, prop={'size': 16}, ncol=2)
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

shots_list = [94652, 94658, 94785, 94792, 94967, 94968, 94972, 94973, 95298, 97405, 97469, 97470, 97476, 97477, 97478, 97828]

for shot in shots_list:
    # Get predictions:
    labelers = glob('my_utime_project_JET_v4_v4_gradual/eval/val_data/files/'+str(shot)+'-*/')
    labeler = labelers[0]
    lab = labeler.split('/')[-2]
    # Get computed kappa value for this labeler
    df = pd.read_csv('my_utime_project_JET_v4_v4_gradual/eval/val_data/evaluation_kappa.csv')
    #print(df[lab])
    df = df.rename(columns={"Unnamed: 0": "labelers"})
    kappa_l = np.round(df.loc[df['labelers'] == lab]['cls 0'].values[0],2)
    kappa_h = np.round(df.loc[df['labelers'] == lab]['cls 1'].values[0],2)
    
    with np.load(labeler+'pred.npz') as data:
        pred = data['arr_0']
    
    # Get PD and times signals
    shots_dir = 'dataset_JET/test'
    shot_file = os.path.join(shots_dir, 'shot_'+lab+'.pkl')
    
    with open(shot_file, 'rb') as f:
        shot_df = pickle.load(f)
    
    PD = shot_df.PD.values
    times = shot_df.time.values
    
    #Plot the whole shot
    #plot_all_signals_all_trans(times, PD, pred, shot, kappa_l, kappa_d, kappa_h, slice_)
    
    # Plot slices (zoom)
    len_shot = PD.shape[0]
    len_seq = 1000
    points_per_window = 10
    
    pred = downsample_labels (pred, points_per_window)
    
    #for i in range(0, len_shot//len_seq):
    slices = np.arange(0, len_shot//len_seq + 1, 3)
    
    for s in range(0,len(slices)-1):
        times_list = []
        PD_list = []
        pred_list = []
    
        for i in range(slices[s], slices[s+1]):
            times_ = times[i*len_seq:(i+1)*len_seq]
            PD_ = PD[i*len_seq:(i+1)*len_seq]
            pred_ = pred[i*len_seq//points_per_window:(i+1)*len_seq//points_per_window]
            times_list.append(times_)
            PD_list.append(PD_)
            pred_list.append(pred_)
            if (s == len(slices)-2):
                last_time_index = (i+1)*len_seq
    
        plot_all_signals_all_trans_slices(times_list, PD_list, pred_list, shot, kappa_l, kappa_h, s, points_per_window)
        
    
    if (len_shot%len_seq):
        #times_ = times[-(len_shot%len_seq):]
        #PD_ = PD[-(len_shot%len_seq):]
        #pred_ = pred[-(len_shot%len_seq)//points_per_window:]
        times_ = times[last_time_index:]
        PD_ = PD[last_time_index:]
        pred_ = pred[last_time_index//points_per_window:]
        
        plot_all_signals_all_trans(times_, PD_, pred_, shot, kappa_l, kappa_h, 'end', points_per_window)
