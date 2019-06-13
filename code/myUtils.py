from collections import defaultdict
from glob import glob
from random import choice, sample

import cv2
import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetLarge
from tqdm import tqdm


def main():
    train_file_path = "../input/train_relationships.csv"
    train_folders_path = "../input/train/"
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

    train = [x for x in relationships if val_famillies not in x[0]]  # train relation dictionary
    val = [x for x in relationships if val_famillies in x[0]]  # test relation dictionary

    #################################
    #########Training Part###########
    #################################
    # for saving the checkpoint
    file_path = "baseline.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # adaptively change the learning rate
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

    callbacks_list = [checkpoint, reduce_on_plateau]

    model = baseline_model()
    # model.load_weights(file_path)
    model.fit_generator(gen_over_sampling(train, train_person_to_images_map, batch_size=16),
                        use_multiprocessing=True,
                        validation_data=gen(val, val_person_to_images_map, batch_size=16),
                        epochs=100,
                        verbose=2,
                        workers=4,
                        callbacks=callbacks_list,
                        steps_per_epoch=200,
                        validation_steps=100)

    #################################
    #########Testing Part############
    #################################
    test_path = "../input/test/"

    submission = pd.read_csv('../input/sample_submission.csv')

    predictions = []

    for batch in tqdm(chunker(submission.img_pair.values)):
        # predict X1[i] and X2[i] relation
        X1 = [x.split("-")[0] for x in batch]  # e.g. ['face00411.jpg', 'face05891.jpg']
        X1 = np.array([read_img(test_path + x) for x in X1])

        X2 = [x.split("-")[1] for x in batch]
        X2 = np.array([read_img(test_path + x) for x in X2])

        pred = model.predict([X1, X2]).ravel().tolist()
        predictions += pred  # predictions.append(pred)

    submission['is_related'] = predictions

    submission.to_csv("baseline.csv", index=False)


def chunker(seq, size=32):
    # A generator.
    # It will cut seq(a list) into lots of pieces, and len(every piece) = size
    # e.g. chunker([1,2,3,4,5],2) = [1,2]->[3,4]->[5]
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def read_img(path, resize=()):
    img = cv2.imread(path)  # scale pixels between -1 and 1
    if resize:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)
    return preprocess_input(img)


def gen_over_sampling(list_tuples, person_to_images_map, batch_size=16, resize_picture=()):
    # Arguments
    #   list_tuples:
    #       relation dictionary
    #   person_to_images_map:
    #       e.g.{'F0124/MID3': ['../input/train/F0124/MID3/P08626_face1.jpg','../input/train/F0124/MID3/P08627_face2.jpg']
    #
    # Returns:
    #   This is a batch generator, whenever it is called, give a batch has form as follow:
    #   [X1, X2]:
    #       [
    #           [person_1's picture, person_2's picture...],
    #           [person_k's picture, person_j's picture...]
    #       ]
    #   labels:
    #       [1,0,.....], where len(labels)=len(X1)=len(X2), labels[0] = 1 means person_1 and k have relation,
    #                       labels[1] = 0, means person_2 and person_j don't have.
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)  # half sample with relation, half not
        labels = [1] * len(batch_tuples)

        # create a batch where annotation 0 means no relation and 1 means has, the data is (p1,p2).
        # Therefore the data-annotation(element of a batch) pair have such form (p1,p2)-0,(p3,p4)-1 .....
        while len(batch_tuples) < batch_size:
            # randomly choose 2 persons' ID
            p1 = choice(ppl)  # person's ID
            p2 = choice(ppl)

            # if 2 persons don't have relation then execute
            # link all persons' without relation together with label 0
            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)
                # after that, labels = [1,1,1....1, 0,0,0,...]
        # if person x[0] doesn't have picture(or directly call doesn't exist), then print his ID
        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        # randomly choose a picture of every person in this batch
        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x, resize_picture) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x, resize_picture) for x in X2])

        yield [X1, X2], labels


def gen(list_tuples, person_to_images_map, batch_size=16, resize_picture=()):
    ppl = list(person_to_images_map.keys())
    while True:
        np.random.shuffle(list_tuples)
        batches = chunker(list_tuples, batch_size // 2)
        for bat in batches:
            labels = [1] * len(bat)
            # create a batch where annotation 0 means no relation and 1 means has, the data is (p1,p2).
            # Therefore the data-annotation(element of a batch) pair have such form (p1,p2)-0,(p3,p4)-1 .....
            while len(bat) < batch_size:
                # randomly choose 2 persons' ID
                p1 = choice(ppl)  # person's ID
                p2 = choice(ppl)
                # if 2 persons don't have relation then execute
                # link all persons' without relation together with label 0
                if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                    bat.append((p1, p2))
                    labels.append(0)
                    # after that, labels = [1,1,1....1, 0,0,0,...]
            # if person x[0] doesn't have picture(or directly call doesn't exist), then print his ID
            for x in bat:
                if not len(person_to_images_map[x[0]]):
                    print(x[0])
            # print(bat)
            # print(labels)
            # randomly choose a picture of every person in this batch
            X1 = [choice(person_to_images_map[x[0]]) for x in bat]
            X1 = np.array([read_img(x, resize_picture) for x in X1])

            X2 = [choice(person_to_images_map[x[1]]) for x in bat]
            X2 = np.array([read_img(x, resize_picture) for x in X2])

            yield [X1, X2], labels, bat


# define model
def baseline_model():
    # keras's input data structure, a tensor
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    # backbone, ResNet50() is a keras function
    # include_top para. = False means cut out the FC layer(the top 3 layer)
    # base_model = NASNetLarge(include_top=False, weights='imagenet')
    base_model = Xception(weights='imagenet', include_top=False)

    # the last 3 layers aren't trainable
    for x in base_model.layers:
        print(x)
        x.trainable = True

    # output(prediction)
    x1 = base_model(input_1)
    x2 = base_model(input_2)

    # x1_ = Reshape(target_shape=(7*7, 2048))(x1)
    # x2_ = Reshape(target_shape=(7*7, 2048))(x2)
    #
    # x_dot = Dot(axes=[2, 2], normalize=True)([x1_, x2_])
    # x_dot = Flatten()(x_dot)

    # Concatenate\GlobalMaxPool2D\GlobalAvgPool2D are keras's function
    # This code block does such things:
    #   average pool and Max pool x1 and pile the 2 pooling results to
    #   create a (224,224,6) tensor
    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x = Multiply()([x1, x2])

    # pile x and x3
    x = Concatenate(axis=-1)([x, x3])

    # Dense layer
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)

    # pack the model to an object and return it
    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    model.summary()

    return model


if __name__ == '__main__':
    main()



