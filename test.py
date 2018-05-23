"""
Descriptions
    "main.py"
References
    # Freeze Weights
        https://stackoverflow.com/questions/35298326/freeze-some-variables-scopes-in-tensorflow-stop-gradient-vs-passing-variables
    # Tensorflow mutiple sessions with multiple GPUs
        https://stackoverflow.com/questions/34775522/tensorflow-multiple-sessions-with-multiple-gpus
    # Saver
        goodtogreate.tistory.com/entry/Saving-and-Restoring
"""

# Public python modules
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

# Custom Pythone modules
import vgg16_adap
import cfmtx
from gen_data import generate
from load_data import load
from snub36_50_category import class_names
from snub36_50_off_category import class_names as class_names2
from cfmtx import cfmtx2


# TODO: set the parameters below
gpu_device = 'device:GPU:1'                     # Which GPU are you going to use?
gen_data = False                                # Generate dataset (.p format)? True or False
fold = 1                                        # which Fold (k) in the metadata is test data?

rec_name = 'result/test' + str(fold) + '.csv'              # Record results into .csv; Name of the .csv file
pretrain_weights = 'saver_weights.npz'                     # Which weights are you going to transfer?
metadata_path = 'dataset/metadata_box_test_0.csv'          # where is meta-data?
testdata_path = 'dataset/test.p'                           # Where is validation data in .p format

label_column_name = 'category'                  # @metdata.csv, Which index indicates category?
batch_size_test = 1                             # Batch size of test data


# With "gpu_device"
with tf.device(gpu_device):
    # If "gen_data" = True, generate a dataset in .p format
    if gen_data:
        generate(metadata_path = metadata_path, data_path = testdata_path,
                 batch_size = batch_size_test, label_column_name=label_column_name, is_training = False, fold=fold)
    else:
        pass

    # Calculate mean of each channel
    #- Load data (.p)
    patch_mean = np.array([0, 0, 0], np.float32)        # Initialize mean of each channel
    dataframe = load(testdata_path, batch_size_test)      # Instance
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

    # Logits, y_out, loss
    logits = vgg.fc4l
    y_out = tf.nn.softmax(logits)

    # Loading validation data (.p)
    dataframe_test = load(testdata_path, batch_size_test)
    num_batch_test = dataframe_test.n_batch

    print("Start test...")

    cfm = np.zeros([len(class_names2), len(class_names)])
    # Test
    for i in range(num_batch_test):
        batch_x, batch_y = dataframe_test.next_batch()
        # - For confusion mtx
        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: batch_x})[0]  # Probability

        # Top 1
        predict_digit = (np.argsort(prob)[::-1])[0]
        predict = class_names[predict_digit]
        label_digit = np.argmax(batch_y)
        label = class_names2[np.argmax(batch_y)]
        print("label:",label_digit, "pred;",predict_digit)

        # Update confusion matrix
        cfm = cfm + cfmtx2(label_digit, predict_digit, cfm.shape)

        # Top 2
        #preds = (np.argsort(prob)[::-1])[0:2]
        #for p in preds:
        #    print(class_names[p], prob[p])
        #print("---------------------------")

    # Save values in the recording variables
    record_cfm = pd.DataFrame(cfm)
    record_cfm.to_csv(rec_name)

    # Draw cfm
    cfmtx.draw2(cfm, normalize=True, xticks_ref=class_names, yticks_ref=class_names2)



