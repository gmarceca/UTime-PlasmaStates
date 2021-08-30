from glob import glob
import os
folds = [1,2,3]
#folds = [1]

epochs = ['01', '03', '05', '07', '09', '11', '13', '15', '17', '19', '21', '23', '25', '27', '29', '31', '33']
#epochs = ['01']

for ep in epochs:
    for fold in folds:
        command = 'python unet_to_cnnLSTM_scores_vs_epochs.py --fold='+str(fold) + ' --epoch=' + str(ep)
        print(command)
        os.system(command)
