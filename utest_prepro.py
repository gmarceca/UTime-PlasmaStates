import sys
sys.path.insert(0, "./utime")
import utime.bin.preprocess_plasma_state_data as prep
prep.entry_func(["--data_dir=dataset_JET", "--machine=JET", "--test=True"])
