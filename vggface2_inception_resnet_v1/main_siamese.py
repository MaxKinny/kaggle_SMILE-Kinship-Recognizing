import tensorflow as tf
import cv2
import numpy as np
from glob import glob
from collections import defaultdict
from random import choice, sample
from myUtils import read_img, gen, chunker
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

###################################################################
#################visualize graph using tensorboard and pb file#####
###################################################################
# This code block is copied from
# /home/public/anaconda3/pkgs/tensorflow-base-1.9.0-gpu_py36h6ecc378_0/
# lib/python3.6/site-packages/tensorflow/python/tools/import_pb_to_tensorboard.py
from tensorflow.python.framework import ops
from tensorflow.python.client import session
# from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

# from tensorflow.python.summary import summary

# model_dir = './model/model.meta'
# log_dir = './logs'
# with session.Session(graph=ops.Graph()) as sess:
#     with tf.gfile.FastGFile(model_dir, "rb") as f:
#       graph_def = graph_pb2.GraphDef()
#       #graph_def.ParseFromString(f.read())
#       importer.import_graph_def(graph_def)
#     pb_visual_writer = tf.summary.FileWriter(log_dir)
#     pb_visual_writer.add_graph(sess.graph)
#     print("Model Imported. Visualize by running: "
#           "tensorboard --logdir={}".format(log_dir))

#####################################################################################
################################Kaggle!!!!!!!########################################
#####################################################################################

#######################################################
#################pre-process#########################
#######################################################
train_file_path = "./input/train_relationships.csv"
train_folders_path = "./input/train/"
val_famillies = "F09"  # use family NO.900 to validate

all_images = glob(
    train_folders_path + "*/*/*.jpg")  # e.g. ['dir/F0002/MID2/P00009_face2.jpg','dir/F0003/MID1/P00010_face3.jpg']

train_images = [x for x in all_images if val_famillies not in x]
val_images = [x for x in all_images if val_famillies in x]  # ['dir/F0009/.../*.jpg']

train_person_to_images_map = defaultdict(list)

ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in
       all_images]  # every person's ID(or directly refer to a person) e.g. ['F0124/MID3', 'F0124/MID3']

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(
        x)  # tell you all this person's pictures e.g.{'F0124/MID3': ['../input/train/F0124/MID3/P08626_face1.jpg','../input/train/F0124/MID3/P08627_face2.jpg']

val_person_to_images_map = defaultdict(list)  # create an empty dict.

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

relationships = pd.read_csv(train_file_path)  # dataFrame
relationships = list(zip(relationships.p1.values,
                         relationships.p2.values))  # e.g. [('F0002/MID1', 'F0002/MID3'), ('F0002/MID2', 'F0002/MID3')]
relationships = [x for x in relationships if
                 x[0] in ppl and x[1] in ppl]  # clean the data, cos some persons may not exist.

train = [x for x in relationships if val_famillies not in x[0]]  # training relation dictionary
val = [x for x in relationships if val_famillies in x[0]]  # testing relation dictionary
# super-parameters
epochs = 200
batch_size = 16

#######################################################
#################define graph #########################
#######################################################
g1 = tf.Graph()
with g1.as_default():
    # pre-define
    def variable_summaries(var):  # it is not used yet
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.squre(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('max', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


    # for v in tf.trainable_variables():
    #     print(v)

    # graph-define
    with tf.name_scope("global_step"):
        global_steps = tf.Variable(0, trainable=False)

    with tf.name_scope("x1"):
        x1 = tf.placeholder(dtype=tf.float32, shape=[None, 160, 160, 3], name='x1')
        with tf.name_scope("pre-trained_model_wreckage"):
            phase_train_input1 = tf.constant(True)
            batch_size_op1 = tf.constant(batch_size)
            learning_rate1 = tf.constant(0, dtype=tf.float32)

    with tf.name_scope("x2"):
        x2 = tf.placeholder(dtype=tf.float32, shape=[None, 160, 160, 3], name='x2')
        with tf.name_scope("pre-trained_model_wreckage"):
            phase_train_input2 = tf.constant(True)
            batch_size_op2 = tf.constant(batch_size)
            learning_rate2 = tf.constant(0, dtype=tf.float32)

    with tf.name_scope("y_label"):
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

    # extractor/reuse variable between model_for_x1 and model_for_x2
    with tf.variable_scope("model", reuse=True) as scope:
        with tf.variable_scope("model_for_x1"):
            saver1 = tf.train.import_meta_graph(
                './model/model.meta',
                input_map={
                    'input': x1,
                    'phase_train': phase_train_input1,
                    'batch_size': batch_size_op1,
                    'learning_rate': learning_rate1})  # 加载模型的grah
            output1 = g1.get_tensor_by_name("model/model_for_x1/InceptionResnetV1/Block8/add:0")

        scope.reuse_variables()

        with tf.variable_scope("model_for_x2"):
            saver2 = tf.train.import_meta_graph(
                './model/model.meta',
                input_map={
                    'input': x2,
                    'phase_train': phase_train_input2,
                    'batch_size': batch_size_op2,
                    'learning_rate': learning_rate2})  # 加载模型的grah
            output2 = g1.get_tensor_by_name("model/model_for_x2/InceptionResnetV1/Block8/add:0")

    with tf.name_scope("x1_output"):
        output1 = tf.identity(output1)

    with tf.name_scope("x2_output"):
        output2 = tf.identity(output2)

    # compute the Euclidean distance between two outputs.
    with tf.name_scope("Distance"):
        with tf.name_scope("pooling_to_vector"):
            flatten_output1_ave = tf.reduce_mean(output1, axis=[1, 2])
            flatten_output1_max = tf.reduce_max(output1, axis=[1, 2])
            flatten_output1 = tf.concat([flatten_output1_ave, flatten_output1_max], -1)

            flatten_output2_ave = tf.reduce_mean(output2, axis=[1, 2])
            flatten_output2_max = tf.reduce_max(output2, axis=[1, 2])
            flatten_output2 = tf.concat([flatten_output2_ave, flatten_output2_max], -1)

        subtract_squre = tf.square(flatten_output1 - flatten_output2, name='subtract_squre')
        multiply = flatten_output1*flatten_output2

        # flatten1 = tf.contrib.layers.flatten(subtract_squre)

        FC1 = tf.layers.dense(  # fully connected layer
            tf.concat([subtract_squre, multiply], -1),
            100,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name='FC1',
            reuse=None)
        dropout = tf.nn.dropout(FC1, keep_prob=1 - 0.01, noise_shape=None, seed=None, name='droput_layer')
        FC2 = tf.layers.dense(  # fully connected layer
            dropout,
            1,
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name='FC2',
            reuse=None)
    # final output
    with tf.name_scope("final_output"):
        logits = tf.identity(FC2, 'logits')
        final_output = tf.layers.dense(  # fully connected layer
            dropout,
            1,
            activation=tf.nn.sigmoid,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name='final_output',
            reuse=None)

    # loss function
    with tf.name_scope("Loss"):
        loss_per = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y,
            logits=logits,
            name='loss_per')
        loss = tf.reduce_mean(loss_per, name='loss_average')

    # Optimizer
    with tf.name_scope("Optimizer"):
        optimizer = tf.train.AdamOptimizer(0.0000005, name='Adam2')  # already has a adam optimizer, so rename it
        train_step = optimizer.minimize(loss,
                                        global_step=global_steps)  # every turn global_steps will automatically plus 1

    var1_list = tf.trainable_variables(scope='model/model_for_x1')
    var2_list = tf.trainable_variables(scope='model/model_for_x2')
    length = len(var1_list)
    i = 0
    assign_op_list = []
    for var1, var2 in zip(var1_list, var2_list):
        assign_op_list.append(tf.assign(var2, var1))
        print('assign!!!1************ %f, total: %f' % (i, length))
        i += 1

# initializer (Replaced By: tf.global_variables_initializer().run())
    # with tf.name_scope("Initializer"):
    #     init = tf.initialize_all_variables()

    # visualization
    # with tf.name_scope("Summary"):
    #     loss_summary = tf.summary.scalar("loss", loss)
    #     merged_summary = tf.summary.merge_all()
#######################################################
#################training part#########################
#######################################################
val_loss_list = []
train_loss_list = []
with tf.Session(graph=g1) as sess:
    train_writer = tf.summary.FileWriter("./logs" + "/train", sess.graph)
    val_writer = tf.summary.FileWriter("./logs" + "/val")
    tf.global_variables_initializer().run()  # initialize the weights
    saver1.restore(sess, './model/model')  # cover the model_for_x1's weights by pre-trained model
    saver2.restore(sess, './model/model')  # cover the model_for_x2's weights by pre-trained model
    test_share_input = np.zeros((1, 160, 160, 3))
    var1_list = tf.trainable_variables(scope='model/model_for_x1')
    length = len(var1_list)
    print("*****************Start Training!!!******************")
    for epochNum in range(epochs):
        for iterNum in range(len(train) // (batch_size//2) + 1):
            train_batch_data, train_annotation = next(gen(train, train_person_to_images_map, batch_size, (160, 160)))
            train_annotation = np.reshape(train_annotation, (batch_size, 1)) * 1.0
            print('**********************************')
            # test whether share or not
            dist, weight_value = sess.run([subtract_squre, var1_list[0]], feed_dict={x1: test_share_input, x2: test_share_input})
            print('test whether share or not, dist: %f, epoch: %d, iter: %d, weight value: %f' % (np.sum(dist), epochNum, iterNum, np.sum(weight_value)))
            # training
            train_los, _ = sess.run([loss, train_step],
                                    feed_dict={
                                        x1: train_batch_data[0],
                                        x2: train_batch_data[1],
                                        y: np.reshape(train_annotation, (batch_size, 1))})
            # train_writer.add_summary(summary, global_steps)
            train_loss_list.append((epochNum, iterNum, train_los))  # record validation loss
            # share the weights
            # for id, op in enumerate(assign_op_list):
            #     sess.run(op)
            #     print('sharing, %f/%f' % (id, length))
            # validate
            valid_batch_data, valid_annotation = next(gen(val, val_person_to_images_map, batch_size, (160, 160)))
            valid_annotation = np.reshape(valid_annotation, (batch_size, 1)) * 1.0
            valid_los = sess.run(loss,
                                 feed_dict={
                                     x1: valid_batch_data[0],
                                     x2: valid_batch_data[1],
                                     y: valid_annotation})
            # val_writer.add_summary(summary, global_steps)
            val_loss_list.append((epochNum, iterNum, valid_los))  # record validation loss
            print('epoch: %d, iteration: %d, train_loss_per_iter: %f, valid_loss_per_epoch: %f' % (epochNum + 1, iterNum + 1, train_los, valid_los))
    train_writer.close()
    val_writer.close()
    #######################################################
    #################Testing Part #########################
    #######################################################
    test_path = "./input/test/"
    submission = pd.read_csv('./input/sample_submission.csv')
    predictions = []
    for batch in tqdm(chunker(submission.img_pair.values)):
        # predict X1[i] and X2[i] relation
        X1 = [x.split("-")[0] for x in batch]  # e.g. ['face00411.jpg', 'face05891.jpg']
        X1 = np.array([read_img(test_path + x, (160, 160)) for x in X1])

        X2 = [x.split("-")[1] for x in batch]
        X2 = np.array([read_img(test_path + x, (160, 160)) for x in X2])

        pred, dist = sess.run([final_output, subtract_squre], feed_dict={x1: X1, x2: X2})
        predictions += pred.flatten().tolist()
    submission['is_related'] = predictions
    submission.to_csv("baseline.csv", index=False)
    print('****************Test whether the model variables are reused or not*************')
    print(sess.run(final_output, feed_dict={x1: np.zeros((1, 160, 160, 3)), x2: np.zeros((1, 160, 160, 3))}))

# out put loss-steps
plt.subplot(1, 2, 1)
plt.plot([tu[0] for tu in val_loss_list], [tu[2] for tu in val_loss_list], 'r*')

plt.subplot(1, 2, 2)
plt.plot([tu[0] for tu in train_loss_list], [tu[2] for tu in train_loss_list], 'b*')



