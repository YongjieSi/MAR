import os
import json
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import pandas as pd
#### the Class of all i* dataset return all the train data with target and test data with target

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None



class filibrispeech(iData):
    use_path = True

    class_order = np.arange(50).tolist()
    def download_data(self):
        # assert 0,"You should specify the folder of your dataset"
        root_dir = '/data/datasets/librispeech_fscil'

        all_train_df = pd.read_csv("/data/caowc/PyCIL/data/librispeech/librispeech_fscil_train.csv")
        all_val_df = pd.read_csv("/data/caowc/PyCIL/data/librispeech/librispeech_fscil_val.csv")
        all_test_df = pd.read_csv("/data/caowc/PyCIL/data/librispeech/librispeech_fscil_test.csv")

        self.train_data = [os.path.join(root_dir, "100spks_segments", all_train_df['filename'][i]) \
                                            for i in range(len(self.class_order)*500)]
        self.train_targets = [all_train_df['label'][i] for i in range(len(self.class_order)*500)]

        self.test_data = [os.path.join(root_dir, "100spks_segments", all_test_df['filename'][i]) \
                                            for i in range(len(self.class_order)*100)]
        self.test_targets = [all_test_df['label'][i] for i in range(len(self.class_order)*100)]