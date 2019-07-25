from collections import defaultdict
from glob import glob
from myUtils import gen2, gen, gen_over_sampling, gen_completely_separated, get_a_fold, stratified_k_fold, read_img
from augmentation import seperation
from sklearn.externals import joblib


import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model

import argparse

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

KTF.set_session(session)


def baseline_model():
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

    model = multi_gpu_model(model, gpus=4)
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))  # default 1e-5

    model.summary()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KFold")
    parser.add_argument('-s', '--SelectedK', default=1)
    parser.add_argument('-c', '--CreateKF', default=True)
    parser.add_argument('-k', '--KNumber', default=5)
    args = parser.parse_args()

    basestr = 'KFold'
    train_file_path = "../input/train_relationships.csv"
    train_folders_path = "../input/train/"

    all_images = glob(train_folders_path + "*/*/*.jpg")
    person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in all_images:
        person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]
    fake_annotation = np.ones(len(relationships))
    if args.CreateKF == "True":
        # cos this dataset doesn't have directly annotations, so create a fake one to feed into stratified_k_fold function.
        print('********************', args.KNumber)
        kfolds_indices = stratified_k_fold(relationships, fake_annotation, int(args.KNumber))
        joblib.dump(kfolds_indices, 'kfold.pkl')  # save the folding results
    else:
        kfolds_indices = joblib.load('kfold.pkl')
        data, _ = get_a_fold(relationships, fake_annotation, kfolds_indices, int(args.SelectedK))
        train = data[0]
        val = data[1]
        # print("Training relationship data:", train)
        # print("Validation relationship data:", val)
        file_path = "vgg_face_" + basestr + ".h5"

        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

        tbCallBack = TensorBoard(log_dir='./logs/' + basestr,
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=True)

        callbacks_list = [checkpoint, reduce_on_plateau, tbCallBack]

        model = baseline_model()
        print("********encoder's model structure********")
        print(model.summary())
        # model.load_weights(file_path)
        # train_relation_tuple_list = seperation(train, person_to_images_map)
        model.fit_generator(gen2(train, person_to_images_map, batch_size=64),
                            use_multiprocessing=True,
                            validation_data=gen(val, person_to_images_map, batch_size=16),
                            epochs=200,
                            verbose=1,
                            workers=8,
                            callbacks=callbacks_list,
                            steps_per_epoch=200,
                            # len(train_relation_tuple_list)//8 + 1,     # len(x_train)//(batch_size) ！！！！！！！！！！！！！
                            validation_steps=10)

        test_path = "../input/test/"


        def chunker(seq, size=32):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))


        from tqdm import tqdm

        submission = pd.read_csv('../input/sample_submission.csv')

        predictions = []

        for batch in tqdm(chunker(submission.img_pair.values)):
            X1 = [x.split("-")[0] for x in batch]
            X1 = np.array([read_img(test_path + x) for x in X1])

            X2 = [x.split("-")[1] for x in batch]
            X2 = np.array([read_img(test_path + x) for x in X2])

            pred = model.predict([X1, X2]).ravel().tolist()
            predictions += pred

        submission['is_related'] = predictions

        submission.to_csv("vgg_face_" + basestr + str(args.SelectedK) + ".csv", index=False)
