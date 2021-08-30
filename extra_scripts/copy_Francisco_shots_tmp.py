from glob import glob
import os

train_list = [53601, 47962, 61021, 31839, 33638, 31650, 31718, 45103, 32592, 30044, 33567, 26383, 52302, 32195, 26386, 59825, 33271, 56662, 57751, 58182, 33188, 30043, 32716, 42197, 33446, 48580, 57103]

val_list = [30268, 61057, 30290, 30197, 43454, 30310, 60097, 32794, 60275, 33942, 33281, 42514, 62744, 30225, 29511, 34010, 31211, 34309, 32911, 31807, 33459, 57218, 32191, 58460, 31554, 30262, 45105]


for run in val_list:
    shots_files = glob('./dataset_plasma_states/cluster_*/*'+str(run)+'*')
    for f in shots_files:
        command = 'cp ' + f + ' ./dataset_plasma_states/3_CV/split_0/val/'
        print(command)
        os.system(command)
