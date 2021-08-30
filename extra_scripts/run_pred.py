from glob import glob
import os
folds = [1,2,3,4,5,6,7, 9, 11, 13]
#folds = [1]

epochs = ['33', '35', '37', '39', '41', '43', '45', '47', '49', '51', '53', '55', '57', '59', '61', '63', '65', '67', '69']
#epochs = ['01']

for ep in epochs:
    for fold in folds:
        if ep == '33' and (fold == 1 or fold == 2):
            print('Continuing')
            continue
        path = glob('model_fold{}/@epoch_{}_*.h5'.format(fold, ep))
        for p in path:
            weights_file_name = p.split('/')[-1]
            command = 'ut evaluate_plasma_states --out_dir eval --data_split val_data --one_shot --overwrite --weights_file_name=' + weights_file_name + ' --fold='+str(fold) + ' --num_GPUs=0 --no_save_true'
            print(command)
            os.system(command)
