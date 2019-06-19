from glob import glob
from collections import defaultdict
import pandas as pd

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


def seperation(relationships, person_to_images_map):
    relation_tuple_list = []
    for re in relationships:
        left_list = person_to_images_map[re[0]]
        right_list = person_to_images_map[re[1]]
        for lpic in left_list:
            for rpic in right_list:
                relation_tuple_list.append((lpic, rpic))
    return relation_tuple_list


def symmetrization(relation_tuple_list):
    mirror = [tu[::-1] for tu in relation_tuple_list]
    symetrical_tuple_list = mirror + relation_tuple_list
    symetrical_tuple_list = set(symetrical_tuple_list)  # delete repeated elements
    symetrical_tuple_list = list(symetrical_tuple_list)
    return symetrical_tuple_list


