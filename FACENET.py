import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf

"""**CARICO IL MODELLO E LO COMPILO**"""

facenet = keras.models.load_model("/models/facenet")

IMAGE_HEIGHT, IMAGE_WIDTH = 160, 160
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, 3) #whc

np.random.seed(42)
tf.random.set_seed(42)

"""**GET THE COMPETITION FACES**"""

MTCNN_SAVE_GPATH = "/OUTPUTS/MTCNN/CROPPED_GALLERY/"
MTCNN_SAVE_QPATH = "/OUTPUTS/MTCNN/CROPPED_QUERY/"

"""**TOTAL EMBEDDER**"""

import PIL
def standardizeImages(image, outputDim):
  
    img = np.asarray(PIL.Image.open(image))
    img = img.astype('float32')
    standard = (img - img.mean()) / img.std()
    standard = tf.expand_dims(standard, axis=0)
    standard = tf.image.resize(standard, outputDim)
    return standard


def generateDataset(folder, modelInputShape):
    filelistdir = sorted(os.listdir(folder))
    dataset = []
    names = []
    for i,image in enumerate(filelistdir):
        if image.split(".")[1] == "png":
          names.append(os.path.join(folder,image))
          standardized = standardizeImages(image=os.path.join(folder,image), 
                                          outputDim=list(modelInputShape))
          dataset.append(standardized)
    dataset = np.array(dataset).astype('float32')

    if len(dataset.shape)>4:
        shapes = dataset.shape
        dataset.resize(shapes[0], shapes[2], shapes[3], shapes[4])
    return dataset, names

resGallery = generateDataset(MTCNN_SAVE_GPATH, (160,160))
resQuery = generateDataset(MTCNN_SAVE_QPATH, (160,160))

def images_to_encodings(images: np.ndarray, model = facenet) -> np.ndarray:
    embedding = model.predict(images, batch_size=32)
    #embedding = embedding / np.linalg.norm(embedding, ord=2)
    return embedding

gallery = images_to_encodings(resGallery[0])
query = images_to_encodings(resQuery[0])

type(resQuery)

namesGallery = resGallery[1]
namesQuery = resQuery[1]
gallery_encodings = dict()
i=0
for el in namesGallery:
  gallery_encodings[el] = gallery[i]
  i+=1

query_encodings = dict()
i=0
for el in namesQuery:
  query_encodings[el] = query[i]
  i+=1

len(gallery_encodings)
len(query_encodings)

"""**GALLERY EMBEDDER**"""

def image_to_encoding(image: np.ndarray, model = facenet) -> np.ndarray:
    image = tf.expand_dims(tf.cast(image, tf.float32), axis=0) #bwhc
    embedding = model.predict(image)
    embedding = embedding / np.linalg.norm(embedding, ord=2)
    return embedding

gallery_encodings = {}
gallery_imgs_paths = {}
for img_path in os.listdir(MTCNN_SAVE_GPATH):
    tmp = MTCNN_SAVE_GPATH + img_path
    try:
        image = tf.cast(tf.image.resize(plt.imread(tmp), size=(IMAGE_WIDTH, IMAGE_HEIGHT)), dtype=tf.float32)
        nome = img_path.split("/")[-1]
        nome = nome.split(".")[0]
        image_en = image_to_encoding(image)
        gallery_encodings[nome] = image_en
        gallery_imgs_paths[nome] = tmp
    except Exception as e:
      print(e, "Errore nell'estensione del file")
      pass

"""**QUERY EMBEDDER**"""

query_encodings = {}
query_imgs_paths = {}
for img_path in os.listdir(MTCNN_SAVE_QPATH):
    tmp = MTCNN_SAVE_QPATH + img_path
    image = tf.cast(tf.image.resize(plt.imread(tmp), size=(IMAGE_WIDTH, IMAGE_HEIGHT)), dtype=tf.float32)
    nome = img_path.split("/")[-1]
    nome = nome.split(".")[0]
    image_en = image_to_encoding(image)
    query_encodings[nome] = image_en
    query_imgs_paths[nome] = tmp

query_imgs_paths.items()

"""**CONTROLLIAMO LE IMMAGINI DI QUERY**"""

def plot_image(image: np.ndarray, title: str, show: bool=False) -> None:
    img = plt.imread(image)
    plt.imshow(img) #se le immagini sono in jpg si VISUALIZZANO con img.astype("uint8")
    plt.title(title)
    plt.axis('off')
    if show:
        plt.show()
plt.figure(figsize=(20,125))
for index, el in enumerate(query_imgs_paths.items()):
  plt.subplot(125,20, index+1)
  plot_image(el[1], el[0])
plt.show

"""**QUI INIZIA LA PARTE CRUCIALE**"""

from numpy import dot
from numpy.linalg import norm
# Calcolo della distanza del coseno tra due vettori
def cosine_distance(vector1, vector2):
    cos_sim = dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    cosine_distance = 1 - cos_sim
    return cosine_distance

from numpy import absolute, sum
# Calcolo della distanza di Manhattan tra due vettori
def manhattan_distance(vector1, vector2):
    manhattan_dist = sum(absolute(vector1 - vector2))
    return manhattan_dist

from numpy import maximum
# Calcolo della distanza di Chebyshev tra due vettori
def chebyshev_distance(vector1, vector2):
    chebyshev_dist = maximum(absolute(vector1 - vector2))
    return chebyshev_dist


risultati=dict()
for image_query in query_encodings:
  print("------")
  print(f"Query:{image_query}")
  fs_query = query_encodings[image_query]
  dists = []
  for image_gallery in gallery_encodings:
    fs_gallery = gallery_encodings[image_gallery]
    dist = manhattan_distance(fs_query, fs_gallery)
    score = (dist, image_gallery)
    dists.append(score)
  dists.sort(key=lambda x: x[0])
  res = []
  for i in range(10):
    res.append(dists)
    print(f"{i}:{dists[i][1]}")
  risultati[image_query] = dists[:10]

for query in risultati.keys():
  image = query_imgs_paths[query]
  plt.figure(figsize=(2,2))
  plt.subplot(1,1,1)
  plt.imshow(plt.imread(image))
  plt.title(query)
  plt.show()
  plt.figure(figsize=(20,8))
  i=0
  for gal in risultati[query]:
      plt.subplot(4, 10, i+1)
      plt.imshow(plt.imread(gallery_imgs_paths[gal[i][1]]))#se le immagini sono in jpg si VISUALIZZANO con img.astype("uint8")
      plt.axis('off')
      plt.title(str(gal[0]))
      i+=1
      if i == 10:
        break
  plt.show()

submitdict = dict()

#submitdict['groupname'] = "The Algorithm Avengers"
for query in risultati:
  vals = risultati[query]
  #res_list = [tup for tup in vals] 
  #res_list = vals
  tmp = []
  for i in range(len(vals)):
    tmp.append(str(vals[i][1].split('_')[0])+'.jpg')   
  
  
  new_res_list = tmp
  #new_res_list = [str(el[1].split("/")[-1][:-2]) + ".jpg" for el in res_list]
  submitdict[str(query.split('_')[0]) + ".jpg"] = new_res_list
  
print(len(submitdict))

mydata = dict()
mydata['groupname'] = "The Algorithm Avengers"
mydata['images'] = submitdict
print(mydata)

import json
import requests

def submit(results, url="https://competition-production.up.railway.app/results/"):
  res = json.dumps(results)
  response = requests.post(url, res)
  try:
    result = json.loads(response.text)
    print(f"accuracy is {result['results']}")
    return result
  except json.JSONDecodeError:
    print(f"ERROR: {response.text}")
    return None
    
submit(mydata)

