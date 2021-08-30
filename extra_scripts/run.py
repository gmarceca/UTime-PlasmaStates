import os

for fold in [1,2,3,4,5]:
    command = "ut train_plasma_states_detector --num_GPUs=1 --fold={}".format(fold)
    print(command)
    os.system(command)

