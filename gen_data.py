# Public python modules
import numpy as np
import pandas as pd
import pickle
import feature
from os import path

class generate():
    def __init__(self, metadata_path, data_path, batch_size, label_column_name, is_training, fold):
        self.batch_size = batch_size
        self.token_stream = []
        self.metadata_path = metadata_path
        self.data_path = data_path
        self.label_column_name = label_column_name
        self.is_training = is_training
        self.fold = fold
        # Run "generate_data()" automatically
        self.generate_data()

    def generate_data(self):
        # meta-data
        meta_df = pd.read_csv(self.metadata_path)

        # Generate training data or validation data?
        if self.is_training:
            meta_df = meta_df[meta_df['fold'] != self.fold]
        else:
            meta_df = meta_df[meta_df['fold'] == self.fold]

        # Collect category names whose "set_split" == 'training' (or "set_split" == 'validation' )
        label_dict = {k: v for v, k in enumerate(sorted(set(meta_df[self.label_column_name].values)))}
        #print(label_dict)

        # Append
        tid_append = []                                         # Audio track ID
        class_append = []                                       # Class
        patch_append = []                                       # Patch

        # Loop
        for i, row in meta_df.iterrows():
            tid = row['track_id']
            label = row[self.label_column_name]
            event_start = row['event_start']
            # Extract patch
            result, patch = feature.feature(tid, event_start)
            # Append
            if result:
                tid_append.append(tid)
                class_append.append(label_dict.get(label))
                patch_append.append(patch)
                print('successfully extracted patch : {}'.format(tid))

        # Write appended array into data frame
        df = pd.DataFrame()
        df['track_id'] = tid_append
        df['category'] = class_append
        df['patch'] = patch_append

        # Shuffle rows (for better training)
        df = df.iloc[np.random.permutation(len(df))]

        self.data_frame = df
        self.num_class = len(label_dict)
        #print(self.num_class)

        # Save "data_frame" as .p (pickle)
        pickle.dump(self.data_frame, open(self.data_path, "wb"))

        # If you want to see the structure of "self.data_frame" (* It is not recommended)
        # print(self.data_frame)

if __name__ == "__main__":

    metadata_path = 'dataset/metadata_5fcv_box.csv'
    traindata_path = 'dataset/train.p'
    validdata_path = 'dataset/valid.p'

    # Test: [traindata_path, True] or [validdata_path, False]?
    # run this code on the terminal
    test = generate(metadata_path, traindata_path, 10, 'category', is_training=True, fold=1)  # An instance
    #test2 = generate(metadata_path, validdata_path, 10, 'category', is_training=False,fold=1)  # An instance


