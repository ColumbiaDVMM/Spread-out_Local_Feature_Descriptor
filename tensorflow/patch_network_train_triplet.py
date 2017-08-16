from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import gc
import patch_reader
import patch_cnn_triplet
import numpy as np
from scipy.spatial import distance
from eval_metrics import ErrorRateAt95Recall
import scipy.io as sio
import pickle
from tqdm import tqdm
from datetime import datetime
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", nargs='?', type=float, default = 0.1,
                    help="Learning rate")

parser.add_argument("--data_dir", nargs='?', type=str, default = '/home/xuzhang/project/Medifor/code/Invariant-Descriptor/data/photoTour/',
                    help="Directory to the dataset")

parser.add_argument("--training", nargs='?', type=str, default = 'notredame',
                    help="Training dataset name")

parser.add_argument("--test", nargs='?', type=str, default = 'liberty',
                    help="Test dataset name")

parser.add_argument("--test_2", nargs='?', type=str, default = 'none',
                    help="Second test dataset name (optional)")

parser.add_argument("--gpu_ind", nargs='?', type=str, default = '0',
                    help="which gpu to use")

parser.add_argument("--margin_0", nargs='?', type=float, default = 1.41,
                    help="Margin for random sampled pairs (discarded)")

parser.add_argument("--margin_1", nargs='?', type=float, default = 0.5,
                    help="Margin for positive pair")

parser.add_argument("--margin_2", nargs='?', type=float, default = 0.7,
                    help="Margin for hard negative")

parser.add_argument("--alpha", nargs='?', type=float, default = 1.0,
                    help="Trade-off parameter between GOR and other loss")

parser.add_argument("--beta", nargs='?', type=float, default = 1.0,
                    help="beta")

parser.add_argument("--num_epoch", nargs='?', type=int, default = 20,
                    help="Number of epoch")

parser.add_argument("--descriptor_dim", nargs='?', type=int, default = 128,
                    help="Number of embedding dimemsion")

parser.add_argument("--batch_size", nargs='?', type=int, default = 128,
                    help="Size of training batch")

parser.add_argument("--patch_size", nargs='?', type=int, default = 64,
                    help="Size of the patch")

parser.add_argument("--loss_type", nargs='?', type=int, default = 0,
                    help="Type of the loss function, 0: triplet loss, 1: N pair loss")

parser.add_argument("--reg_type", nargs='?', type=int, default = 0,
                    help="Number of embedding dimemsion")

parser.add_argument("-f", "--full_dataset", action="store_true",
                    help="Trained with full dataset")

parser.add_argument("-s", "--save_dis_mat", action="store_true",
                    help="Save distance of the test pairs")

parser.add_argument("-n", "--no_normalization", action="store_true",
                    help="Don't perform l2 normalization")

args = parser.parse_args()

# Parameters
start_learning_rate = args.learning_rate
num_epoch = args.num_epoch
display_step = 100
batch_size =args.batch_size
num_epoch = int(args.num_epoch*128/batch_size)

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ind

#log suffix
now = datetime.now()
if args.no_normalization:
    suffix = 'triplet_LR{:1.0e}_alpha{:1.1e}_beta{:1.1e}_margin_{:1.1f}_lossType_{}_nn_'.format(start_learning_rate,args.alpha, args.beta, args.margin_1, args.loss_type) \
        +  now.strftime("%Y%m%d-%H%M%S")
else:
    suffix = 'triplet_LR{:1.0e}_alpha{:1.1e}_beta{:1.1e}_margin_{:1.1f}_embedding_{}_lossType_{}_regType_{}_'.format(start_learning_rate, args.alpha, args.beta, args.margin_1,\
            args.descriptor_dim, args.loss_type, args.reg_type) + now.strftime("%Y%m%d-%H%M%S")
        
patch_size = args.patch_size
channel_num = 1
test_batch_size = 1000

descriptor_dim = args.descriptor_dim

train = patch_reader.SiameseDataSet(args.data_dir)
train.load_by_name(args.training, patch_size = patch_size)

test = patch_reader.SiameseDataSet(args.data_dir)
test.load_by_name(args.test, patch_size = patch_size)

print('Loading training stats:')
try:
    file = open('../data/stats_%s.pkl'%args.training, 'r')
    mean, std = pickle.load(file)
except:
    mean, std = train.generate_stats()
    pickle.dump([mean,std], open('../data/stats_%s.pkl'%args.training,"wb"));
print('-- Mean: %s' % mean)
print('-- Std:  %s' % std)
train.normalize_data(mean, std)
test.normalize_data(mean, std)

# get patches
patches_train = train._get_patches()
patches_test  = test._get_patches()
if args.loss_type == 0:
    train.generate_triplet(1000000)
elif args.loss_type == 1:
    train.generate_npair(1000000)

# get matches for evaluation
test_matches  = test._get_matches()
train_matches  = train._get_matches()

CNNConfig = {
    "patch_size": patch_size,
    "margin_0": args.margin_0,
    "margin_1": args.margin_1,
    "margin_2": args.margin_2,
    "descriptor_dim" : descriptor_dim,
    "batch_size" : batch_size,
    "alpha" : args.alpha,
    "beta": args.beta,
    "loss_type": args.loss_type,
    "reg_type": args.reg_type,
    "no_normalization": args.no_normalization,
    "channel_num" : channel_num
}

print('Learning Rate: {}'.format(args.learning_rate))
print('Random Negative Margin: {}'.format(args.margin_0))
print('Positive Margin: {}'.format(args.margin_1))
print('Feature_Dim: {}'.format(args.descriptor_dim))
print('Alpha: {}'.format(args.alpha))

cnn_model = patch_cnn_triplet.PatchCNN_triplet(CNNConfig)

global_step = tf.Variable(0, trainable=False)
decay_step  = 100000
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                           decay_step, 0.96, staircase=True)

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum = 0.9).minimize(cnn_model.cost,global_step=global_step)

# model saver
saver = tf.train.Saver()
model_ckpt = '../model_triplet/'+args.training+'model.ckpt'

# log dir
training_logs_dir = '../tensorflow_log/train_{}/'.format(suffix)
test_logs_dir = '../tensorflow_log/test_{}/'.format(suffix)

# Initializing the variables
init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
# Launch the graph
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    step = 1
    training_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_logs_dir, sess.graph)
    merged = tf.summary.merge_all()
    
    # Keep training until reach max iterations
    for i in range(num_epoch):
        epoch_loss = 0
        num_batch_in_epoch = train.num_train_patch//batch_size
        for step in tqdm(xrange(num_batch_in_epoch)):
            if args.loss_type == 0:
                index_1, index_2, index_3 = train.next_batch_triplet(batch_size)
            elif args.loss_type == 1:
                index_1, index_2, index_3 = train.next_batch_triplet(batch_size)
            
            sess.run(optimizer, feed_dict={cnn_model.patch: patches_train[index_1],\
                    cnn_model.patch_p: patches_train[index_2], \
                    cnn_model.patch_n: patches_train[index_3]})
            
            if step % display_step == 0:
                step = step+1
                if args.loss_type == 0:
                    index_1, index_2, index_3 = train.next_batch_triplet(batch_size)
                elif args.loss_type == 1:
                    index_1, index_2, index_3 = train.next_batch_npair(batch_size)
                fetch = {
                    "cost": cnn_model.cost,
                    "eucd_p": cnn_model.eucd_p,
                }
                summary, result = sess.run([merged, fetch], feed_dict={cnn_model.patch: patches_train[index_1],\
                     cnn_model.patch_p: patches_train[index_2], 
                     cnn_model.patch_n: patches_train[index_3]})

                training_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                
                epoch_loss = epoch_loss+result["cost"]

        epoch_loss = epoch_loss/num_batch_in_epoch*display_step

        offset = 0
        dists  = np.zeros(train_matches.shape[0],)
        labels = np.zeros(train_matches.shape[0],)
        bad_num_1 = 0
        pos_bad_1 = 0
        for x in tqdm(xrange(train_matches.shape[0] // test_batch_size)):
            # get data batch
            batch = train_matches[offset:offset + test_batch_size, :]
            fetch = {
                "cost": cnn_model.cost,
                "eucd_p": cnn_model.eucd_p,
            }
            result = sess.run(fetch, feed_dict={cnn_model.patch: patches_train[batch[:,0]],\
                    cnn_model.patch_p: patches_train[batch[:,1]], 
                    cnn_model.patch_n: patches_train[batch[:,1]]})

            dists[offset:offset + test_batch_size] = result['eucd_p']
            labels[offset:offset + test_batch_size] = batch[:,2]
            offset = offset+test_batch_size
        
        # compute the false positives rate
        fpr95 = ErrorRateAt95Recall(labels, dists)
        fpr_summary = tf.Summary(value=[tf.Summary.Value(tag="fpr95", simple_value=fpr95)])
        training_writer.add_summary(fpr_summary, tf.train.global_step(sess, global_step))
        print("Epoch " + str(i) + ", Val FPR = " +
              "{:.6f}".format(fpr95))

        offset = 0
        dists  = np.zeros(test_matches.shape[0],)
        labels = np.zeros(test_matches.shape[0],)
        for x in tqdm(xrange(test_matches.shape[0] // test_batch_size)):
           # get data batch
            batch = test_matches[offset:offset + test_batch_size, :]
            fetch = {
                "cost": cnn_model.cost,
                "eucd_p": cnn_model.eucd_p,
            }
            result = sess.run(fetch, feed_dict={cnn_model.patch: patches_test[batch[:,0]],\
                    cnn_model.patch_p: patches_test[batch[:,1]],
                    cnn_model.patch_n: patches_test[batch[:,1]]})

            dists[offset:offset + test_batch_size] = result['eucd_p']
            labels[offset:offset + test_batch_size] = batch[:,2]
            offset = offset+test_batch_size

        if args.save_dis_mat and (i+1)%5 == 0:
            save_object = np.zeros((2,), dtype=np.object)
            save_object[0] = dists
            save_object[1] = labels
            sio.savemat('../dis_mat/{}_{}_step_{}.mat'.format(args.training, args.test,i+1), {'save_object':save_object})

        #compute the false positives rate
        fpr95 = ErrorRateAt95Recall(labels, dists)
        fpr_summary = tf.Summary(value=[tf.Summary.Value(tag="fpr95", simple_value=fpr95)])
        test_writer.add_summary(fpr_summary, tf.train.global_step(sess, global_step))
        print("Epoch " + str(i) + ", Test FPR = " +
              "{:.6f}".format(fpr95))

    if args.test_2 != 'none':
        test = []
        patches_test = []
        test_matches = []
        n = gc.collect()
        
        test = patch_reader_new.SiameseDataSet('/home/xuzhang/project/Medifor/code/Invariant-Descriptor/data/photoTour/')
        test.load_by_name(args.test_2, patch_size = patch_size)
        test.normalize_data(mean, std)
        # get patches
        patches_test  = test._get_patches()
        test_matches  = test._get_matches()

        offset = 0
        dists  = np.zeros(test_matches.shape[0],)
        labels = np.zeros(test_matches.shape[0],)
        for x in tqdm(xrange(test_matches.shape[0] // test_batch_size)):
           # get data batch
            batch = test_matches[offset:offset + test_batch_size, :]
            fetch = {
                "cost": cnn_model.cost,
                "eucd_p": cnn_model.eucd_p,
            }
            result = sess.run(fetch, feed_dict={cnn_model.patch: patches_test[batch[:,0]],\
                    cnn_model.patch_p: patches_test[batch[:,1]],
                    cnn_model.patch_n: patches_test[batch[:,1]]})

            dists[offset:offset + test_batch_size] = result['eucd_p']
            labels[offset:offset + test_batch_size] = batch[:,2]
            offset = offset+test_batch_size

        if args.save_dis_mat and (i+1)%5 == 0:
            save_object = np.zeros((2,), dtype=np.object)
            save_object[0] = dists
            save_object[1] = labels
            sio.savemat('../dis_mat/{}_{}_step_{}.mat'.format(args.training, args.test_2, i+1),\
                    {'save_object': save_object})

        #compute the false positives rate
        fpr95 = ErrorRateAt95Recall(labels, dists)
        print("Test Set " + args.test_2 + ", Test FPR = " +
              "{:.6f}".format(fpr95))

    saver.save(sess, model_ckpt)
