import cv2
import tensorflow as tf
import numpy as np
import os
import cv2
import pickle
import base64
import json
from numpy import dot
from numpy.linalg import norm
from pre_processing import Processing
from elasticsearch import Elasticsearch
import dlib
from PIL import Image
import matplotlib.pyplot as plt
import compress as c


def store_data(access_point):
    base_path = 'C:/Users/Harsha/Desktop/tinyfile_system/models/storage/data/'
    filepathToCompress = 'C:/Users/Harsha/Desktop/tinyfile_system/tests/'
    objectCompress = 'test4'
    extension = '.jpg'
    folder_name = access_point
    newPath = base_path + folder_name
    os.chdir(newPath)
    c.compress(objectCompress + '_'+ 'compressed.tar.gz', [filepathToCompress + objectCompress + extension])
    return newPath
    pass

def create_folder(en, es, index_name, path_key, access_point):
    base_path = 'C:/Users/Harsha/Desktop/tinyfile_system/models/storage/data/'
    folder_name = access_point
    try:
        os.mkdir(base_path + folder_name)
        print("Directory '% s' created" % directory)

    except Exception as e:
        #I am avoiding exception handling for the review
        pass

    pass


def access_point(en, es, index_name, path_key):
    pickle_in = open(path_key,"rb")
    facial_vector = pickle.load(pickle_in)
    query_doc = {
      "size" : 1,
      "query": {
        "script_score": {
          "query" : {
            "match_all" : {}
          },
          "script": {
              
            "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 0.5",
            "params": {
              "query_vector": facial_vector
            }
          }
        }
      }
    }
    sentences = []
    search_result = es.search(index = index_name, body = query_doc)
    source = search_result['hits']['hits'][0]['_source']
    doc_id = search_result['hits']['hits'][0]['_id']
    return doc_id
    pass



def delete_index(es, name):
    es.indices.delete(index = name)


def cosine_similarity(a, b):
    cos_sim = np.inner(a, b) / (norm(a) * norm(b))
    return cos_sim

def encoding(en , root_image_path):
    new_root_image_path= 'C:/Users/Harsha/Desktop/root_test_'  + 'root' + '.jpg'
    image = Processing(root_image_path)
    image.resize_save(new_root_image_path)
    x = cv2.imread(new_root_image_path)
    updated_image = cv2.resize(x, (128, 128),interpolation = cv2.INTER_NEAREST)
    new_image = np.expand_dims(updated_image, axis=0)
    encode = en.predict(new_image)[0]
    return encode



def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

def similarity(en):
    matrix = [[0]*5 for i in range(5)]
    for i in range(5):
        root_image_path = 'C:/Users/Harsha/Desktop/tinyfile_system/tests/test' + str(i) + '.jpg'
        x = encoding(en, root_image_path)
        #print(len(x))
        for j in range(5):
            new_image_path = 'C:/Users/Harsha/Desktop/tinyfile_system/tests/test' + str(j) + '.jpg'
            y = encoding(en, new_image_path)
            matrix[i][j] = cosine_similarity(x, y)
    for mat in matrix:
        print(mat)
    return -1
            
def insert_image_test(en, es, name):
    for i in range(5):
        base = 'C:/Users/Harsha/Desktop/tinyfile_system/tests'
        root_image_path = base + '/test' + str(i) + '.jpg'
        temp = cv2.imread(root_image_path, 1)
        image = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        detected_faces = detect_faces(image)
        for n, face_rect in enumerate(detected_faces):
            face = Image.fromarray(image).crop(face_rect)
            new_cropped_path = base + '/buffer.jpg'
            face.save(new_cropped_path)
            x = encoding(en, new_cropped_path)


            meta_data = root_image_path
            document = {"embedding_vector": x, "meta_data": meta_data}
            es.index(index = name, body = document)
    pass

def generate_key(en, es, name):
    base = 'C:/Users/Harsha/Desktop/tinyfile_system/tests'
    key_path = 'C:/Users/Harsha/Desktop/tinyfile_system/'
    root_image_path = base + '/test' + '0.jpg'
    temp = cv2.imread(root_image_path, 1)
    image = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    detected_faces = detect_faces(image)
    for n, face_rect in enumerate(detected_faces):
        face = Image.fromarray(image).crop(face_rect)
        new_cropped_path = base + '/buffer.jpg'
        face.save(new_cropped_path)
        x = encoding(en, new_cropped_path)
        pickle_out = open(key_path + 'key.pickle', 'wb')
        pickle.dump(x, pickle_out)
        pickle_out.close()
        meta_data = root_image_path
        document = {"embedding_vector": x, "meta_data": meta_data}
        es.index(index = name, body = document)
    pass



es = Elasticsearch(HOST = "http://localhost", PORT = 9200)
en = tf.keras.models.load_model('enc.h5', compile=False)
es = Elasticsearch()
index_name = 'nis_review'
path_key = 'C:/Users/Harsha/Desktop/tinyfile_system/key.pickle'

access_point = access_point(en, es, index_name, path_key)

print(store_data(access_point))

'''
create_folder(en, es, index_name, path_key, access_point)
generate_key(en, es, name)
print(similarity(en))
'''

"""
delete_index(es, name)
"""



