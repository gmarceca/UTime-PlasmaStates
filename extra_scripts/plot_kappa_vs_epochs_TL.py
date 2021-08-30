import matplotlib.pyplot as  plt
from glob import glob
import numpy as np

def get_kappa_from_exp(fold):

    path = glob('./eval/val_data_fold{}/files_*/kappa_scores_fold_{}_epoch_*.txt'.format(fold, fold))
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

    exp1 = 1 
    exp2 = 2
    exp3 = 3

    colors = ['blue', 'red', 'green']
    # TL
    for i, ex in enumerate([exp3, exp2, exp1]):
        epochs, L_score, D_score, H_score, Avg_score = get_kappa_from_exp(ex)
        idx = np.argsort(epochs)
        epochs = np.array(epochs)[idx]
        L_score = np.array(L_score)[idx]
        D_score = np.array(D_score)[idx]
        H_score = np.array(H_score)[idx]
        Avg_score = np.array(Avg_score)[idx]

        plt.plot(epochs, Avg_score, color=colors[i], linestyle='--')

    #exp1 = split_to_exp_notl[3]
    #exp2 = split_to_exp_notl[4]
    #exp3 = split_to_exp_notl[5]
    ## NOTL
    #for i, ex in enumerate([exp3, exp2, exp1]):
    #    epochs, L_score, D_score, H_score, Avg_score = get_kappa_from_exp(ex)
    #    idx = np.argsort(epochs)
    #    epochs = np.array(epochs)[idx]
    #    L_score = np.array(L_score)[idx]
    #    D_score = np.array(D_score)[idx]
    #    H_score = np.array(H_score)[idx]
    #    Avg_score = np.array(Avg_score)[idx]

    #    plt.plot(epochs, Avg_score, color=colors[i])

    plt.yticks(np.arange(0, 1, 0.1))

    plt.xlim([0, 105])
    plt.ylim([0, 1])

    plt.xlabel('epochs')
    plt.ylabel('Kappa')
    plt.title('UTime JET')
    plt.legend(('TL 6-shots','TL 3-shots', 'TL 1-shots'), loc = 'best')

    plt.savefig('UTime_TL_kappa_scores_s1.png')
    pass

if __name__ == '__main__':
    main()
