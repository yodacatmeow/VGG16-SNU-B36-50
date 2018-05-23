"""
Descriptions
    "tsne_conv.py"
References

"""

# Public python modules
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Custom Pythone modules
import vgg16_adap
import cfmtx
from gen_data import generate
from load_data import load
from snub36_50_category_num import class_names
from cfmtx import cfmtx2


## TODO: set the parameters below
gpu_device = 'device:GPU:1'                                     # Which GPU are you going to use?
learning_rate = 10                                              # Learning rate
gen_data = False                                                 # Generate dataset (.p format)
run_tsne = False                                                 # Run tSNE
save_tsne = False                                                # save tSNE?
fold = 1                                                        # Which Fold(k) will be used as a data?
rec_name = 'result/tsne_conv.csv'                               # Record "xs", "ys", and "label" as "rec_name".csv
pretrain_weights = 'saver_tr_nf_mspdb_2048_2048_592_k1.npz'     # Which weights are you going to transfer?
metadata_path = 'dataset/metadata_5fcv_box.csv'                 # where is meta-data?
data_path = 'dataset/data.p'                                    # Where is data in .p format
batch_size = 1                                                  # Batch size
label_column_name = 'category'                                  # @metdata.csv, Which index indicates category?


# With "gpu_device"
with tf.device(gpu_device):
    # If "gen_data" = True, generate a dataset in .p format
    if gen_data:
        #- Training data(k="fold") -> .p
        generate(metadata_path = metadata_path, data_path = data_path,
                 batch_size = batch_size, label_column_name=label_column_name, is_training = True, fold=fold)
    else:
        pass

    # Calculate mean of each channel
    #- Load data (.p)
    patch_mean = np.array([0, 0, 0], np.float32)        # Initialize mean of each channel
    dataframe = load(data_path, batch_size)    # Instance
    #- Calculate mean of each channel
    for i, row in dataframe.dataframe.iterrows():
        patch = row['patch']
        patch_mean[0] += np.mean(patch[:, :, 0])
        patch_mean[1] += np.mean(patch[:, :, 1])
        patch_mean[2] += np.mean(patch[:, :, 2])
    patch_mean = patch_mean / len(dataframe.dataframe['patch'])
    #- Delete "dataframe" from memory
    dataframe.left = None

    # Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Placeholders
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])  # [None, width_VGG16 * height_VGG16 * depth_VGG16]

    # VGG16 instance; Transfer the pretraining weights
    vgg = vgg16_adap.vgg16(imgs, patch_mean, pretrain_weights, sess)

    # Load validation data (.p)
    dataframe_test = load(data_path, batch_size)
    num_batch_test = dataframe_test.n_batch

    print("Start test...")

    # Test
    feature = []
    label = []
    # A lot of RAM required (about 8GB)
    for i in range(num_batch_test):
        batch_x, batch_y = dataframe_test.next_batch()
        # - For confusion mtx
        conv_val = sess.run(vgg.conv5_3, feed_dict={vgg.imgs: batch_x})
        conv_val = np.reshape(conv_val, int(conv_val.shape[0] * conv_val.shape[1] * conv_val.shape[2] * conv_val.shape[3]))
        label_num = class_names[np.argmax(batch_y)]
        feature.append(conv_val)
        label.append(label_num)

    # Initialize "xs" and "ys"
    xs = []
    ys = []
    # tSNE
    if run_tsne:
        model = TSNE(learning_rate=learning_rate)
        transformed = model.fit_transform(feature)
        xs = transformed[:, 0]
        ys = transformed[:, 1]
        marker = label
    else:
        # Read  csv file
        df = pd.read_csv(rec_name, sep=',')
        xs = df['xs']
        ys = df['ys']
        marker = df['label']

    # Save results
    if save_tsne:
        record = pd.DataFrame()
        record['xs'] = xs
        record['ys'] = ys
        record['label'] = label
        record.to_csv(rec_name)
    else:
        pass

    # Draw a scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys, c=marker)

    for i, txt in enumerate(marker):
        ax.annotate(txt, (xs[i], ys[i]))

    plt.show()
