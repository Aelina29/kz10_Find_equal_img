import pickle
import PIL.Image
import cv2 as cv
import pandas as pd
import base64
from sklearn.neighbors import NearestNeighbors
import numpy as np
import streamlit as st
import PIL

@st.cache(allow_output_mutation=True)
def load_model():
    loaded_model = pickle.load(open('pic_model.sav', 'rb'))
    return loaded_model

@st.cache(allow_output_mutation=True)
def load_bases():
    im_db = './imagesBaseDf.csv'
    vec_db = './vectorsBaseDf.csv'
    img_base = pd.read_csv(im_db)
    vec_base = pd.read_csv(vec_db)
    vec_base['embedding'] = vec_base['embedding'].apply(
        lambda x: np.frombuffer(base64.b64decode(bytes(x[2:-1], encoding='ascii')), dtype=np.int32))
    return img_base, vec_base

# def encode_image(image: PIL.Image.Image, clf: sklearn.cluster.KMeans) -> np.ndarray:
#     """
#     Encode image to embedding representation
#     :param image: PIL image
#     :param clf: k-means instance from sk-learn lib
#     :return: embedding vector with float32 values
#     """
#
#     _img_arr = np.array(image.convert('RGB'))
#     _, des = sift.detectAndCompute(_img_arr)    # Nx128
#     _classes = clf.predict(np.array(des, dtype=np.float32))
#     emb, _ = np.histogram(_classes, len(clf.cluster_centers_), normed=True)
#     return emb

def encode_image(arr) -> np.ndarray:
    emds = []
    for i in arr:
        emb, _ = np.histogram(i, 2048, normed=True)
        emds.append(emb)
    return emds

def find_indices_in_db(img, classifier, db, db_paths, db_neighbours):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    prediction = classifier.predict(des)
    predHist, _ = np.histogram(prediction, 2048, normed=True)
    distances, indices = db_neighbours.kneighbors(predHist.reshape(1, -1), return_distance=True)
    return db_paths.loc[indices[0], ['img_path']].values, distances[0]

def main():
    st.title('Our Super Search')
    up_file = st.file_uploader('Choice file', type=['jpeg', 'jpg', 'webp', 'png', 'tiff'])
    if up_file is not None:
        try:
            img = PIL.Image.open(up_file)
            img = PIL.ImageOps.exif_transpose(img)
            k = 5
            img_paths, dists = find_indices_in_db(np.array(img)[:, :, ::-1], loaded_model, database, path_database, db_neighbours)
            for i in range(len(img_paths)):
                st.image(PIL.Image.open(img_paths[i][0]),
                         caption='Image {} with dist {}'.format(i + 1, f'{dists[i]:.3f}', width=580))
        except Exception as e:
            st.write('CRASHED:{}'.format(e))

# main()
loaded_model = load_model()
path_database, database = load_bases()
db_neighbours = NearestNeighbors(n_neighbors=5, metric='cosine')
emdbs = encode_image(database['embedding'].values)
db_neighbours.fit(emdbs)
if __name__ == '__main__':
    main()