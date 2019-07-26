from collections import defaultdict
from glob import glob
from random import choice, sample
from myUtils import gen, gen_over_sampling, gen_completely_separated, read_img
from augmentation import seperation
import logging
import cv2
import os
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import nni

LOG = logging.getLogger('mnist_keras')
K.set_image_data_format('channels_last')
TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']


class SendMetrics(keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)
        nni.report_intermediate_result(logs["val_acc"])


def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'optimizer': 'Adam',
        'learning_rate': 0.001
    }


def baseline_model(hyper_params):
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model1 = VGGFace(model='resnet50', include_top=False, name="vggface_resnet50_leg1")
    base_model2 = VGGFace(model='resnet50', include_top=False, name="vggface_resnet50_leg2")

    for x in base_model1.layers[:-3]:
        x.trainable = True

    for x in base_model2.layers[:-3]:
        x.trainable = True


    x1 = base_model1(input_1)
    x2 = base_model2(input_2)

    # x1_ = Reshape(target_shape=(7*7, 2048))(x1)
    # x2_ = Reshape(target_shape=(7*7, 2048))(x2)
    #
    # x_dot = Dot(axes=[2, 2], normalize=True)([x1_, x2_])
    # x_dot = Flatten()(x_dot)

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x = Multiply()([x1, x2])

    x = Concatenate(axis=-1)([x, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    if hyper_params['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(lr=hyper_params['learning_rate'])
    else:
        optimizer = keras.optimizers.SGD(lr=hyper_params['learning_rate'], momentum=0.9)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=optimizer)  # default 1e-5

    model.summary()

    return model


def train(params):
    basestr = 'splitmodel'
    train_file_path = "../input/train_relationships.csv"
    train_folders_path = "../input/train/"
    val_famillies = "F09"

    all_images = glob(train_folders_path + "*/*/*.jpg")

    train_images = [x for x in all_images if val_famillies not in x]
    val_images = [x for x in all_images if val_famillies in x]

    train_person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)

    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

    train = [x for x in relationships if val_famillies not in x[0]]
    val = [x for x in relationships if val_famillies in x[0]]

    file_path = "vgg_face_" + basestr + ".h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

    # tbCallBack = TensorBoard(log_dir='./logs/'+basestr,
    #                          histogram_freq=0,
    #                          write_graph=True,
    #                          write_images=True)


    callbacks_list = [SendMetrics(), TensorBoard(log_dir=TENSORBOARD_DIR), checkpoint, reduce_on_plateau]

    model = baseline_model(params)
    # model.load_weights(file_path)
    # train_relation_tuple_list = seperation(train, train_person_to_images_map)
    model.fit_generator(gen(train, train_person_to_images_map, batch_size=16),
                        use_multiprocessing=True,
                        validation_data=gen(val, val_person_to_images_map, batch_size=16),
                        epochs=50,
                        verbose=2,
                        workers=4,
                        callbacks=callbacks_list,
                        steps_per_epoch=200, # len(train_relation_tuple_list)//8 + 1,     # len(x_train)//(batch_size) ！！！！！！！！！！！！！
                        validation_steps=10)
    x_test, y_test = next(gen(val, val_person_to_images_map, batch_size=16))
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    nni.report_final_result(acc)

if __name__ == '__main__':
    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_default_params()
        PARAMS.update(RECEIVED_PARAMS)
        # train
        train(PARAMS)
    except Exception as e:
        LOG.exception(e)
        raise
# test_path = "../input/test/"
#
#
# def chunker(seq, size=32):
#     return (seq[pos:pos + size] for pos in range(0, len(seq), size))
#
#
# from tqdm import tqdm
#
# submission = pd.read_csv('../input/sample_submission.csv')
#
# predictions = []
#
# for batch in tqdm(chunker(submission.img_pair.values)):
#     X1 = [x.split("-")[0] for x in batch]
#     X1 = np.array([read_img(test_path + x) for x in X1])
#
#     X2 = [x.split("-")[1] for x in batch]
#     X2 = np.array([read_img(test_path + x) for x in X2])
#
#     pred = model.predict([X1, X2]).ravel().tolist()
#     predictions += pred
#
# submission['is_related'] = predictions
#
# submission.to_csv("vgg_face_"+basestr+".csv", index=False)
