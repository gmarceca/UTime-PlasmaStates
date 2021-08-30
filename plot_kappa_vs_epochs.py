import matplotlib.pyplot as  plt
from glob import glob
import numpy as np

def get_kappa_from_exp(name, fold, dir_):

    path = glob(dir_+'/{}/val_data_fold{}/files_*/kappa_scores_fold_{}_epoch_*.txt'.format(name, fold, fold))
    L_score = []
    D_score = []
    H_score = []
    Avg_score = []
    epochs = []
    for p in path:
        ep = int(p.split('/')[-1].split('_')[-1][:-4])
        epochs.append(ep)
        f = open(p, "r")
        vals = f.read()[1:-2].split(' ')
        vals = list(filter(None, vals))
        L_score.append(float(vals[0]))
        D_score.append(float(vals[1]))
        H_score.append(float(vals[2]))
        Avg_score.append(float(vals[3]))
        pass
    return epochs, L_score, D_score, H_score, Avg_score

def main():
    # Dictionary mapping split_* to experiment number
    #split_to_exp_tl = {0:7, 1:8, 2:9, 3:10, 4:11, 5:12, 6:13, 8:15, 10:17, 12:19}
    #split_to_exp_notl = {0:20, 1:21, 2:22, 3:23, 4:24, 5:25, 6:26, 8:27, 10:28, 12:29}

    epochs, L_score, D_score, H_score, Avg_score_TCV = get_kappa_from_exp('eval_TCV', '1', 'TCV_detrend')
    epochs, L_score, D_score, H_score, Avg_score_JET = get_kappa_from_exp('eval_JET', '1', 'TCV_detrend')
    idx = np.argsort(epochs)
    epochs = np.array(epochs)[idx]
    Avg_score_TCV = np.array(Avg_score_TCV)[idx]
    Avg_score_JET = np.array(Avg_score_JET)[idx]

    plt.plot(epochs, Avg_score_TCV, color='blue', linestyle='-')
    plt.plot(epochs, Avg_score_JET, color='red', linestyle='-')

    plt.yticks(np.arange(0, 1, 0.1))

    horiz_line_data = np.arange(-1, 100, 1)
    plt.plot(horiz_line_data, np.ones(len(horiz_line_data))*0.63, 'r--')

    plt.xlim([-1, 110])
    plt.ylim([0, 1])

    plt.xlabel('epochs')
    plt.ylabel('Kappa')
    plt.title('UTime trained in TCV')
    plt.legend(('val set TCV', 'val set JET', 'without detrending'), loc = 'best')

    plt.savefig('UTime_TCV_model_kappa_scores_vs_epochs.png')
    pass

if __name__ == '__main__':
    main()
