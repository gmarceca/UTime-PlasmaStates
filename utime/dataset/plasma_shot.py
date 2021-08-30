"""
Implements the PlasmaShot class which represents a plasma discharge
"""
import os
import numpy as np
from contextlib import contextmanager
import pickle
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from utime.bin.helper_funcs import *
from glob import glob

class PlasmaShot(object):
    """
    Represents a plasma discharge object
    """

    def __init__(self, shot_id, 
            subject_dir,
            cfg_dic,
            states=['L', 'D', 'H'],
            load=False,
            debug=False):
        
        '''
        shot_id: 'ShotNumber-labeler', i.e: 3211-ffelici
        subject_dir: Path to stored input files
        cfg_dic: cfg file dataset dictionary
        states: plasma states to load (for the labels only)
        '''
        self.debug = debug
        self.subject_dir = subject_dir
        self.states = cfg_dic['states']
        self.shot_id = shot_id

        if len(self.states) == 3:
            self.states_dic = {'L':0, 'D': 1, 'H': 2}
        elif len(self.states) == 2:
            self.states_dic = {'L':0, 'H': 1}
        
        self.time_spread = cfg_dic['seq_length']
        self.time_spread = self.time_spread + 10
        self.pad_seq = cfg_dic['pad_seq']
        self.points_per_window = cfg_dic['points_per_window']
        self.diagnostics = cfg_dic['diagnostics']
        self.project_dir = cfg_dic['project_dir']
        self.read_csv = cfg_dic['read_csv']
        self.validate_score = cfg_dic['validate_score']
        self.machine = cfg_dic['Machine']

        self.plasma_shot_df = None
        self.list_IDs = defaultdict(list)
        self._plasma_shot = None
        self._states = None
        self.crop = None
        
        if load:
            self.load()
        

    @property
    def plasma_shot(self):
        """ Returns the PS object (an ndarray of shape [-1, n_channels]) """
        return self._plasma_shot

    @property
    def identifier(self):
        """
        Returns shot ID: "shot-labeler"
        """
        return self.shot_id[:-4] if not self.read_csv else self.shot_id
    
    @property
    def class_to_period_dict(self):
        return {c: np.where(self.plasma_states == self.states_dic[c])[0] for c in self.states}

    def __len__(self):
        """ Returns the size of the PlasmaShot = periods * data_per_period """
        return self._plasma_shot.shape[1] * self.points_per_window

    @property
    def plasma_states(self):
        """ Returns the plasma state labels """
        return self._states
    
    @property
    def plasma_times(self):
        return self._times

    def unload(self):
        """ Unloads the PlasmaShot and states data """
        self._plasma_shot = None
        self._states = None

    def reload(self, warning=True):
        """ Unloads and loads """
        if warning and self.loaded:
            print("Reloading PlasmaShot '{}'".format(self.identifier))
        self.unload()
        self.load()


    def get_all_periods(self):
        """
        Returns the full (dense) data of the PlasmaShot

        Returns:
            X: An ndarray of shape [self.n_periods,
                                    self.data_per_period,
                                    self.n_channels]
            y: An ndarray of shape [self.n_periods, 1]
        """
        X = self._plasma_shot.reshape(-1, self.data_per_period, self.n_channels)
        y = self.plasma_states
        if len(X) != len(y):
            err_msg = ("Length of PlasmaShot array does not match length dense "
                       "states array ({} != {}) ".format(len(X),len(y)))
            self.raise_err(ValueError, err_msg)
        return X, y

    @contextmanager
    def loaded_in_context(self):
        """ Context manager from automatic loading and unloading """
        self.load()
        try:
            yield self
        finally:
            self.unload()

    @property
    def loaded(self):
        """ Returns whether the PlasmaShot data is currently loaded or not """
        return not any((self.plasma_shot is None,
                        self.plasma_states is None))

    @property
    def data_per_period(self):
        """
        Computes and returns the data (samples) per period of
        'period_length_sec' seconds of time (en 'epoch' in sleep research)
        """
        return self.points_per_window

    @property
    def n_channels(self):
        """ Returns the number of channels in the PSG array """
        return len(self.diagnostics)

    @property
    def n_sample_channels(self):
        """
        Returns the number of channels that will be returned by
        self.extract_from_psg (this may be different from self.n_channels if
        self.channel_sampling_groups is set).
        """
        return self.n_channels

    def load_shot_pkl (self):

        with open(os.path.join(self.subject_dir, 'shot_{}'.format(self.shot_id)), 'rb') as f:
            self.plasma_shot_df = pickle.load(f)

    def load_shot_csv (self):
        if self.validate_score:
            self.plasma_shot_df = pd.read_csv(glob(os.path.join(self.subject_dir, self.machine + '_'  + self.shot_id + '_*_labeled.csv'))[0], encoding='utf-8')
        else:
            self.plasma_shot_df = pd.read_csv(os.path.join(self.subject_dir, self.machine + '_'  + self.shot_id + '_signals.csv'), encoding='utf-8')
        self.ori_plasma_shot_df = self.plasma_shot_df
    
    def preprocess_shot (self):
        if self.machine == 'TCV':
            self.plasma_shot_df = remove_current_30kA(self.plasma_shot_df)
        elif self.machine == 'JET':
            self.plasma_shot_df = remove_current_1MA(self.plasma_shot_df)
        if self.validate_score:
            self.plasma_shot_df = remove_no_state(self.plasma_shot_df) # This is only possible when shot is Validated
        self.plasma_shot_df = remove_disruption_points(self.plasma_shot_df)
        self.plasma_shot_df = self.plasma_shot_df.reset_index(drop=True)
        self.intersect_times = np.round(self.plasma_shot_df.time.values,5)
        self.plasma_shot_df = self.plasma_shot_df[self.plasma_shot_df['time'].round(5).isin(self.intersect_times)]
        self.plasma_shot_df = normalize_signals_mean(self.plasma_shot_df, self.machine)

    def reshape_ (self, df_signal, indexes, points_per_window):
        ret = np.asarray([df_signal]).reshape(int(len(indexes)/points_per_window), points_per_window)
        return ret

    def concatenate_channels_and_reshape (self):
        # Crop the signal in order to be an exact multiple of points_per_window
        if self.crop:
            self.plasma_shot_df = self.plasma_shot_df[:-self.crop]
            if self.read_csv:
                self.ori_plasma_shot_df = self.ori_plasma_shot_df[:-self.crop]
                self.intersect_times = self.intersect_times[:-self.crop]
        if (self.plasma_shot_df.shape[0] % self.points_per_window is not 0):
            ValueError("sequence length {} is not divisible by points_per_window {}".format(self.plasma_shot_df.shape[0], self.points_per_window))
        
        self._plasma_shot = np.empty((1, self.plasma_shot_df.shape[0]//self.points_per_window, self.points_per_window, self.n_channels))
        if self.pad_seq and self.plasma_shot_df.shape[0] < self.time_spread:
                self._plasma_shot_padded = np.empty((1, self.time_spread//self.points_per_window, self.points_per_window, self.n_channels))
        for i, d in enumerate(self.diagnostics):
            sig = self.plasma_shot_df[d].values
            self._plasma_shot[0,:,:,i] = self.reshape_ (sig, np.arange(0, self.plasma_shot_df.shape[0]), self.points_per_window)
            if self.pad_seq and self.plasma_shot_df.shape[0] < self.time_spread:
                    self._plasma_shot_padded[0,:,0,i] = np.pad(self._plasma_shot[0,:,0,i], (0, (self.time_spread-self.plasma_shot_df.shape[0])//self.points_per_window), 'constant')
        
        if self.pad_seq and self.plasma_shot_df.shape[0] < self.time_spread:
            self._plasma_shot = self._plasma_shot_padded

        self._times = self.plasma_shot_df.time.values
        
    def get_labels (self):
        """
        """
        labels = self.plasma_shot_df.LHD_label.values.tolist()
        states = []
        for k in range(int(len(labels)/self.points_per_window)):
            window = labels[k*self.points_per_window : (k+1)*self.points_per_window]
            # determine the label of a window by getting the dominant class
            label = max(set(window), key = window.count)
            # L mode: label = 1, D mode: label = 2, H mode: label = 3
            # rest one so label is 0, 1 or 2
            label -= 1
            states.append(label)

        self._states = np.asarray([states]).swapaxes(0, 1)    


    def _load(self):
        
        if self.read_csv:
            self.load_shot_csv()
            self.preprocess_shot()
        else:
            self.load_shot_pkl()
        
        if self.plasma_shot_df.shape[0]%self.points_per_window is not 0:
            self.crop = self.plasma_shot_df.shape[0]%self.points_per_window

        # preprocess inputs and store them (from df to np arrays)
        self.concatenate_channels_and_reshape()
        
        if not self.read_csv:
            self.get_labels()
        else:
            # Just set it with zero values
            self._states = np.zeros(self._plasma_shot.shape[1])

    def load(self):
        """
        High-level function invoked to load the PlasmaShot data
        """
        if not self.loaded:
            try:
                self._load()
            except Exception as e:
                raise errors.CouldNotLoadError("Unexpected load error for sleep "
                                               "study {}. Please refer to the "
                                               "above traceback.".format(self.identifier),
                                               study_id=self.identifier) from e
        return self
