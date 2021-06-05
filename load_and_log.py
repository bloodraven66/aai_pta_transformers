import numpy as np
import scipy.io
import os
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import time
import random
import librosa


def load_data(mode, subject_id, subjects, precompute=False):

        #load articulator data, phoneme sequence and durations for entire data in experimental setup for your dataset.
        Youtval, phon_valseq, phon_val_dur, val_lengths, Youttrain, phon_trainseq, phon_train_dur, train_lens
