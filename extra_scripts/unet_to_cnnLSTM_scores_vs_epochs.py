import numpy as np
from glob import glob
import os
from scipy import stats
from argparse import ArgumentParser

def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Evaluate a U-Time model.')
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold number")
    parser.add_argument("--epoch", type=str, default='01',
                        help="Epoch number")
    return parser

def calc_mode(values):
    assert len(values.shape) == 2
    assert values.shape[0] > values.shape[1] #if not, incoming array axes must be swapped!
    if values.shape[1] == 1:
        return np.squeeze(values)
    modes = []
    for v_id, v in enumerate(values):
        # print(v)
        if len(np.unique(v)) == values.shape[1]:
            #FIXME
            modes += [-1,]
        else:
            # print(stats.mode(v))
            modes += [stats.mode(v)[0][0]]
    # print(modes)
    modes = np.asarray(modes)
    modes = np.squeeze(modes)
    return modes

def dice_coefficient(predicted_states, labeled_states):
    total_low_intersects = 0
    total_low_state_trues = 0
    total_low_state_positives = 0
    total_high_intersects = 0
    total_high_state_trues = 0
    total_high_state_positives = 0
    total_dither_intersects = 0
    total_dither_state_trues = 0
    total_dither_state_positives = 0
    # predicted_states += 1 #no none state
    # print(len(predicted_states), len(labeled_states))
    assert(len(predicted_states) == len(labeled_states))

    # none_state_positives = np.zeros(len(predicted_states))
    # none_state_positives[predicted_states == 0] = 1
    low_state_positives = np.zeros(len(predicted_states))
    low_state_positives[predicted_states == 1] = 1
    # print('here', np.sum(low_state_positives))
    dither_state_positives = np.zeros(len(predicted_states))
    dither_state_positives[predicted_states == 2] = 1
    high_state_positives = np.zeros(len(predicted_states))
    high_state_positives[predicted_states == 3] = 1
    # a,b,c,d = np.sum(none_state_positives),
    a,b,c, = np.sum(low_state_positives), np.sum(dither_state_positives), np.sum(high_state_positives)
    # print('posit', a, b, c, a+b+c)

    # none_state_trues = np.zeros(len(labeled_states))
    # none_state_trues[labeled_states == 0] = 1   
    low_state_trues = np.zeros(len(labeled_states))
    low_state_trues[labeled_states == 1] = 1
    # plt.plot(labeled_states)
    # plt.show()
    dither_state_trues = np.zeros(len(labeled_states))
    dither_state_trues[labeled_states == 2] = 1

    high_state_trues = np.zeros(len(labeled_states))
    high_state_trues[labeled_states == 3] = 1
    # print(sum(dither_state_trues))
    # exit(0)
    # a,b,c,d = np.sum(none_state_trues),
    a,b,c = np.sum(low_state_trues), np.sum(dither_state_trues), np.sum(high_state_trues)
    # print('trues',  a, b, c, a+b+c)

    # none_intersect_cardinality = np.sum(np.logical_and(none_state_positives, none_state_trues))
    # total_none_intersects += none_intersect_cardinality
    # total_none_state_trues += np.sum(none_state_trues)
    # total_none_state_positives += np.sum(none_state_positives)  

    low_intersect_cardinality = np.sum(np.logical_and(low_state_positives, low_state_trues))
    total_low_intersects += low_intersect_cardinality
    total_low_state_trues += np.sum(low_state_trues)
    total_low_state_positives += np.sum(low_state_positives)
    # low_dsc = (2.*low_intersect_cardinality)/(np.sum(low_state_trues) + np.sum(low_state_positives))

    dither_intersect_cardinality = np.sum(np.logical_and(dither_state_positives, dither_state_trues))
    total_dither_intersects += dither_intersect_cardinality
    total_dither_state_trues += np.sum(dither_state_trues)
    total_dither_state_positives += np.sum(dither_state_positives)
    # print np.sum(intersect_dither)
    # dither_dsc = (2.*dither_intersect_cardinality)/(np.sum(dither_state_trues) + np.sum(dither_state_positives))

    high_intersect_cardinality = np.sum(np.logical_and(high_state_positives, high_state_trues))
    total_high_intersects += high_intersect_cardinality
    total_high_state_trues += np.sum(high_state_trues)
    total_high_state_positives += np.sum(high_state_positives)
    # high_dsc = (2.*high_intersect_cardinality)/(np.sum(high_state_trues) + np.sum(high_state_positives))

    # a,b,c, d= total_none_state_positives, total_low_state_positives, total_dither_state_positives, total_high_state_positives
    a,b,c = total_low_state_positives, total_dither_state_positives, total_high_state_positives
    # print('positives', a, b, c, a+b+c)
    # a,b,c, d= none_intersect_cardinality, low_intersect_cardinality, dither_intersect_cardinality, high_intersect_cardinality
    a,b,c = low_intersect_cardinality, dither_intersect_cardinality, high_intersect_cardinality
    # print('itsct', a, b, c, a+b+c)

    # if(total_none_state_trues + total_none_state_positives) > 0:
    #     none_dsc = (2.*total_none_intersects)/(total_none_state_trues + total_none_state_positives)
    # else:
    #     none_dsc = 1
    if(total_low_state_trues + total_low_state_positives) > 0:
        low_dsc = (2.*total_low_intersects)/(total_low_state_trues + total_low_state_positives)
    else:
        low_dsc = 1
    if(total_dither_state_trues + total_dither_state_positives) > 0:
        dither_dsc = (2.*total_dither_intersects)/(total_dither_state_trues + total_dither_state_positives)
    else:
        # print('dither dsc')
        # print(total_dither_state_trues)
        # print(total_dither_state_positives)
        dither_dsc = 1
    if(total_high_state_trues + total_high_state_positives) > 0:
        high_dsc = (2.*total_high_intersects)/(total_high_state_trues + total_high_state_positives)
    else:
        high_dsc = 1

    # s_nst = sum(none_state_trues)
    s_lst = sum(low_state_trues)
    s_hst = sum(high_state_trues)
    s_dst = sum(dither_state_trues)
    # none_state_trues_pc = s_nst/len(labeled_states)
    low_state_trues_pc = s_lst/len(labeled_states)
    high_state_trues_pc = s_hst/len(labeled_states)
    dither_state_trues_pc = s_dst/len(labeled_states)
    # total_dsc = none_dsc * none_state_trues_pc + low_dsc*low_state_trues_pc + high_dsc*high_state_trues_pc + dither_dsc*dither_state_trues_pc

    total_dsc = low_dsc*low_state_trues_pc + high_dsc*high_state_trues_pc + dither_dsc*dither_state_trues_pc
    # print('Calc of mean val for dice', low_dsc, low_state_trues_pc, high_dsc, high_state_trues_pc, dither_dsc, dither_state_trues_pc, total_dsc)
    return np.asarray([low_dsc, dither_dsc, high_dsc, total_dsc])


def k_statistic(predicted, labeled):
    k_index = []
    state_trues_pc = []
    for i, state in enumerate(['L', 'D', 'H']):
        s = i + 1
        assert(len(predicted) == len(labeled))
        predicted_states = np.zeros(len(predicted))
        predicted_states[predicted == s] = 1
        labeled_states = np.zeros(len(labeled))
        labeled_states[labeled == s] = 1


        # if np.array_equal(labeled_states, np.zeros(len(labeled))) and sum(predicted_states) < 2:
        #     predicted_states = np.zeros(len(predicted))
        # 

        s_tot = sum(labeled_states)
        state_trues_pc += [round(s_tot/len(labeled_states),3)]
        yy = 0
        yn = 0
        ny = 0
        nn = 0
        total = len(predicted_states)
        for k in range(total):
            if predicted_states[k] == 1 and labeled_states[k] == 1:
                yy += 1
            elif predicted_states[k] == 0 and labeled_states[k] == 0:
                nn += 1
            elif predicted_states[k] == 1 and labeled_states[k] == 0:
                ny += 1
            elif predicted_states[k] == 0 and labeled_states[k] == 1:
                yn += 1
        assert(yy + yn + ny + nn == total)
        p0 = (yy + nn) /  total
        pyes = ((yy + yn) / total) * ((yy + ny) / total)
        pno = ((ny + nn) / total) * ((yn + nn) / total)
        pe = pyes + pno
        # print('ps....................', p0, pyes, pno, pe)
        if pe == 1:
            k_index +=[1]
        else:
            score = (p0 - pe) / (1-pe)
            if score < 0: #careful, as k-statistic can actually be smaller than 0!
                score = 0
            k_index += [score]
    mean_k_ind = np.average(np.asarray(k_index), weights=state_trues_pc)
    return np.asarray(k_index + [mean_k_ind])


def calc_consensus(values):
    assert len(values.shape) == 2
    assert values.shape[0] > values.shape[1] #if not, incoming array axes must be swapped!
    if values.shape[1] == 1:
        return np.squeeze(values)
    consensus_labels = []
    # print(values.shape)
    # print('computing modes')
    for v_id, v in enumerate(values):
        vals = np.unique(v)[0]
        if len(np.unique(v)) == 1: #if there is a single label
            consensus_labels += [vals]
        else:
            consensus_labels += [-1,]

    consensus_labels = np.asarray(consensus_labels)
    consensus_labels = np.squeeze(consensus_labels)
    return consensus_labels


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


def run(args):
    states_pred_concat =[]
    ground_truth_concat = []
    k_indexes =[]
    k_indexes_dic = {}
    consensus_concat = []
    points_per_window = 10
    
    from utime.hyperparameters import YAMLHParams
    project_dir = os.path.abspath('./')
    
    if bool(args.fold):
        hpath = project_dir + "/dataset_configurations/dataset_1_fold{}.yaml".format(str(args.fold))
    else:
        hpath = project_dir + "/dataset_configurations/dataset_1.yaml"
    
    dataset_hparams = YAMLHParams(hpath)
    
    shots = {int(s.split('/')[-1].split('_')[1][:5]) for s in glob(os.path.join(dataset_hparams['val_data']['data_dir'], 'shot*'))}
   
    save_path = './eval/val_data_fold{}/files_ep{}/'.format(args.fold, args.epoch)

    for i, shot in enumerate(shots):
        
        if bool(args.fold):
            labelers = glob('eval/val_data_fold{}/files_ep{}/'.format(args.fold, args.epoch)+str(shot)+'-*/')
        else:
            labelers = glob('eval/val_data/files/'+str(shot)+'-*/')
    
        labeler_states = []
        pred_states_disc = []
    
        for k, labeler in enumerate(labelers):
    
            # Read stored predictions and GT from UNet eval
            with np.load(labeler+'pred.npz') as data:
                pred = data['arr_0']
            with np.load(labeler+'true.npz') as data:
                true = data['arr_0']
        
            labeler_states += [true]
        
        print('FOLD:::::::::::::: ', args.fold)
        print('EPOCH:::::::::::::: ', args.epoch)
        print('SHOT:::::::::::::: ', shot)
    
        labeler_states = np.asarray(labeler_states)
        pred_states_disc += [pred]
        pred_states_disc = np.asarray(pred_states_disc)
        pred_states_disc = pred_states_disc.T
        
        pred_states_disc = np.squeeze(pred_states_disc)
        
        if (labeler_states.shape[2] == 1):
            labeler_states = np.squeeze(labeler_states, 2)
        

        # Downsampling labels and predictions
        pred_states_disc = downsample_labels (pred_states_disc, points_per_window)
        labeler_states = downsample_labels (np.squeeze(labeler_states), points_per_window)
        labeler_states = labeler_states.reshape(labeler_states.shape[0], 1)
        labeler_states = labeler_states.T

        pred_states_disc += 1 #necessary because argmax returns 0 to 2, while we want 1 to 3!
        labeler_states += 1 #necessary because argmax returns 0 to 2, while we want 1 to 3!
    
        assert(labeler_states.shape[1] == pred_states_disc.shape[0])
        
        states_pred_concat.extend(pred_states_disc)
    
        ground_truth = calc_mode(labeler_states.swapaxes(0,1))
        ground_truth_concat.extend(ground_truth)
        
        dice_cf = dice_coefficient(pred_states_disc, ground_truth)
        k_st = k_statistic(pred_states_disc, ground_truth)
        k_indexes += [k_st]
        #print('dice: ', dice_cf)
        print('kst: ', k_st)
        k_indexes_dic[shot] = k_st
        
        consensus = calc_consensus(labeler_states.swapaxes(0,1)) #has -1 in locations which are not consensual, ie at least one person disagrees (case 3)
        consensus_concat.extend(consensus)
    
    
    k_indexes = np.asarray(k_indexes)
    
    ground_truth_concat = np.asarray(ground_truth_concat)
    consensus_concat = np.asarray(consensus_concat)
    states_pred_concat = np.asarray(states_pred_concat)
    
    score_with_consensus = True
    
    if score_with_consensus:
        ground_truth_mask = np.where(ground_truth_concat!=-1)[0]
        
        ground_truth_concat = ground_truth_concat[ground_truth_mask]
        states_pred_concat = states_pred_concat[ground_truth_mask]
        consensus_concat = consensus_concat[ground_truth_mask] #should stay the same, as consensus is subset of ground truth

    avg_kappa = k_statistic(states_pred_concat, ground_truth_concat)
    avg_dice = dice_coefficient(states_pred_concat, ground_truth_concat)
    with open(save_path + 'kappa_scores_fold_{}_epoch_{}.txt'.format(args.fold, args.epoch), 'w') as f:
        f.write(np.array2string(avg_kappa))

    print('Averaged dice predictions: ', avg_dice)
    print('Averaged kappa predictions: ', avg_kappa)
    print('Averaged kappa labelers: ', k_statistic(consensus_concat, ground_truth_concat))
 

def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    run(parser.parse_args(args))

if __name__ == "__main__":
    entry_func()

