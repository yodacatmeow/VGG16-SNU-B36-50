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
from gen_data import generate
from load_data import load
from cfmtx import cfmtx


# TODO: set the following parameters
gpu_device = 'device:GPU:0'                     # Which GPU are you going to use?
is_transfer_learn = True                        # Transfer the pre-trained weights?
gen_data = True                                 # Generate dataset (.p format)?
freeze_layer = False                            # Are you going to freeze certain layers? (Check optimizer code below)
bn = False                                      # Turn on batch normalization?
saver = False                                    # Are you going to save whole weights after training?
fold = 1                                        # Fold k; k-fold cross-validation

rec_name = 'result/tr_nf_mspdb_2048_2048_592' + str(fold)                   # Save results as .csv;
rec_name_cfm = 'result/cfm_tr_nf_mspdb_2048_2048_592' + str(fold) + 'ep'    # Record confusion matrix as .csv
pretrain_weights = 'vgg16_weights.npz'                                      # Where is the pretraining weights?
saver_name = 'saver_tr_nf_mspdb_2048_2048_592_k5.npz'                       # if saver = True, save the weights as .npz
metadata_path = 'dataset/metadata_5fcv_box.csv'                             # where is meta-data?
traindata_path = 'dataset/train_5fcv_k' + str(fold) + '.p'                  # Where is training data?
validdata_path = 'dataset/valid_5fcv_k' + str(fold) + '.p'                  # Where is validation data?

label_column_name = 'category'                  # In the metadata, which index indicates category?
n_category = 39                                 # The number of categories;
batch_size_tr = 39                              # Batch size of training data
batch_size_val = 39                             # Batch size of validation data
n_epoch = 50                                    # Epoch
learning_rate = 0.001                           # Learning rate

# With "gpu_device"
with tf.device(gpu_device):
    # If "gen_data" = True, generate dataset in .p format
    if gen_data:
        generate(metadata_path = metadata_path, data_path = traindata_path,
                            batch_size = batch_size_tr, label_column_name=label_column_name,
                            is_training = True, fold=fold)
        generate(metadata_path = metadata_path, data_path = validdata_path,
                            batch_size = batch_size_tr, label_column_name=label_column_name,
                            is_training = False, fold=fold)
    else:
        pass

    # Calculate mean of each channel
    #- Load the training data (.p); Note that "dataframe" is an instance
    patch_mean = np.array([0, 0, 0], np.float32)                    # Init.
    dataframe = load(traindata_path, batch_size_tr)                 # Instance
    for i, row in dataframe.dataframe.iterrows():
        # Calculate mean of each channel
        patch = row['patch']
        patch_mean[0] += np.mean(patch[:, :, 0])    # Ch 0
        patch_mean[1] += np.mean(patch[:, :, 1])    # Ch 1
        patch_mean[2] += np.mean(patch[:, :, 2])    # Ch 2
        #print(patch_mean)
    patch_mean = patch_mean / len(dataframe.dataframe['patch'])
    print("patch_mean:", patch_mean)
    dataframe.left = None                                           # Delete "dataframe" from the memory

    # Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Placeholders
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])          # [None, width_VGG16 * height_VGG16 * depth_VGG16]
    y = tf.placeholder(tf.float32, [None, n_category])

    # VGG16 instance; Transfer the pretrained weights
    if is_transfer_learn:
        vgg = vgg16_adap.vgg16(imgs=imgs, img_mean=patch_mean, weights=pretrain_weights, sess=sess, bn=bn, bn_is_training=False)
    else:
        vgg = vgg16_adap.vgg16(imgs=imgs, img_mean=patch_mean, sess=sess, bn=bn, bn_is_training=True)

    # Logits, y_out, loss
    logits = vgg.fc4l
    y_out = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    # Accuracy measurement
    correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Optimization
    if freeze_layer:
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, var_list=[vgg.fc3w, vgg.fc3b, vgg.fc4w, vgg.fc4b])
    # Update all layers
    else:
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # One should initialize "FCb" graph for TensorFlow
    # In the case of not transferring the pre-trained weights, we need to initialize the whole graph
    if is_transfer_learn:
        init_new_vars_op = tf.variables_initializer([vgg.fc4w, vgg.fc4b])               # New; New FC layer @"vgg16_adap" needs graph initialization
        sess.run(init_new_vars_op)                                                      # New; Run graph initialization
    else:
        sess.run(tf.global_variables_initializer())

    # Training and validation

    # Variables used for recording training and validation
    rec_epoch = []
    rec_train_err = []
    rec_train_acc = []
    rec_valid_err = []
    rec_valid_acc = []
    rec_cfm = []        # For recording confusion matrix
    rec_epoch_cfm = 0

    print("Start training...")

    # Load the training data (.p); Note that "dataframe" is an instance
    dataframe_tr = load(traindata_path, batch_size_tr)
    num_batch_tr = dataframe_tr.n_batch
    # Load the validation data (.p)
    dataframe_valid = load(validdata_path, batch_size_val)
    num_batch_valid = dataframe_valid.n_batch

    # Loop; iter = epoch
    for epoch in range(n_epoch):
        # Variables for calculating average error and average accuracy
        aver_train_err = 0
        aver_train_acc = 0
        # bn_is_training
        vgg.bn_is_training = True
        # Mini-batch training
        for i in range(num_batch_tr):
            batch_x, batch_y = dataframe_tr.next_batch()
            err, acc, _ = sess.run([loss, accuracy, train_op],
                                   feed_dict={vgg.imgs: batch_x, y: batch_y})
            aver_train_err += err
            aver_train_acc += acc
        aver_train_err = aver_train_err / num_batch_tr
        aver_train_acc = aver_train_acc / num_batch_tr
        print("epoch:", epoch, "av_tr_err:", aver_train_err, "av_tr_acc:", aver_train_acc)

        # Variables for calculating average-error and average-accuracy
        aver_valid_err = 0
        aver_valid_acc = 0
        cfm = np.zeros([n_category, n_category])    # Initialize a confusion matrix
        # bn_is_training
        vgg.bn_is_training = False
        # Mini-batch validation
        for i in range(num_batch_valid):
            batch_x, batch_y = dataframe_valid.next_batch()
            err, acc = sess.run([loss, accuracy], feed_dict={vgg.imgs: batch_x, y: batch_y})

            #- For confusion mtx
            prob = sess.run(vgg.probs, feed_dict={vgg.imgs: batch_x})   # Probability
            preds = (np.argmax(prob, axis = 1))                         # Predictions
            label = (np.argmax(batch_y, axis=1))                        # Labels
            cfm = cfm + cfmtx(label, preds, n_category, batch_size_val) # Update confusion matrix

            aver_valid_err += err
            aver_valid_acc += acc
        aver_valid_err = aver_valid_err / num_batch_valid
        aver_valid_acc = aver_valid_acc / num_batch_valid
        print("epoch:", epoch, "av_val_err:", aver_valid_err, "av_val_acc:", aver_valid_acc)

        # Record via appending
        rec_epoch.append(epoch)
        rec_train_err.append(aver_train_err)
        rec_train_acc.append(aver_train_acc)
        rec_valid_err.append(aver_valid_err)
        rec_valid_acc.append(aver_valid_acc)

    # Save weights
    if saver:
        vgg.save_weights(saver_name, sess)
    else:
        pass

    # Save values in the recording variables
    record = pd.DataFrame()
    record['epoch'] = rec_epoch
    record['train_err'] = rec_train_err
    record['train_acc'] = rec_train_acc
    record['valid_err'] = rec_valid_err
    record['valid_acc'] = rec_valid_acc
    record.to_csv(rec_name)

    # Save the confusion matrix
    record_cfm = pd.DataFrame(cfm)
    record_cfm.to_csv(rec_name_cfm + str(n_epoch))
