import tensorflow as tf
import math


def build_t_model(trajectories):
    """
    Function to build a subgraph
    """
    h1_t_units = 1
    h2_t_units = 1
    M = 1

    with tf.variable_scope('h1_t'):
        weights = tf.get_variable('weights', shape=[])
        biases = tf.get_variable('biases', shape=[])
        h1_t = trajectories*weights + biases

    with tf.variable_scope('h2_t'):
        weights = tf.get_variable('weights', shape=[])
        biases = tf.get_variable('biases', shape=[])
        h2_t = h1_t*weights + biases

    with tf.variable_scope('h3_t'):
        weights = tf.get_variable('weights', shape=[])
        biases = tf.get_variable('biases', shape=[])
        h3_t = h2_t*weights + biases

    return h3_t


g1 = tf.Graph()
with g1.as_default():
    input1 = tf.placeholder(dtype=tf.float32, name="input1")
    input2 = tf.placeholder(dtype=tf.float32, name="input2")
    with tf.variable_scope('traj_embedding') as scope:
        with tf.name_scope("leg1"):
            embeddings_left = build_t_model(input1)
        scope.reuse_variables()
        with tf.name_scope("leg2"):
            embeddings_right = build_t_model(input2)
    # for id, v in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='traj_embedding')):
    #     print((id, v))

with tf.Session(graph=g1) as sess:
    train_writer = tf.summary.FileWriter("./test/", sess.graph)
    # tf.global_variables_initializer().run()
    for var in tf.trainable_variables(scope='traj_embedding'):
        if var._trainable:
            sess.run(tf.assign(var, 0))
        else:
            continue
    w, v = sess.run([embeddings_left, embeddings_right], feed_dict={input1: 0, input2: 0})