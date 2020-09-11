import numpy as np
import scipy.sparse as sp

from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.engine import Model
from sklearn.neighbors import NearestNeighbors
from glob import glob


def vectorize(path, model):
    img = image.load_img(path, target_size=(224, 224))
    # Конвертация PIL image в numpy array (вектор)
    x = image.img_to_array(img)
    # В вектор-строку (2-dims)
    x = np.expand_dims(x, axis=0)
    # Библиотечная подготовка изображения
    x = preprocess_input(x)
    vec = model.predict(x).ravel()
    return vec


def vectorize_all(files, model, px=224, n_dims=512):
    min_idx = 0
    max_idx = len(files)
    preds = sp.lil_matrix((len(files), n_dims))

    X = np.zeros(((max_idx - min_idx), px, px, 3))
    # Каждой картинке ссответствует одна строка X
    i = 0
    for i in range(min_idx, max_idx):
        file = files[i]
        try:
            img = image.load_img(file, target_size=(px, px))
            img_array = image.img_to_array(img)
            X[i - min_idx, :, :, :] = img_array
        except Exception as e:
            print(e)
    max_idx = i
    X = preprocess_input(X)
    these_preds = model.predict(X)
    shp = ((max_idx - min_idx) + 1, n_dims)
    preds[min_idx:max_idx + 1, :] = these_preds.reshape(shp)
    return preds


if __name__ == '__main__':
    images_glob_path = f'data/images/*.jpg'
    files = glob(images_glob_path)

    base_model = VGG19(weights='imagenet')
    # Срезается посл6дний fully connected слой
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    # Использование метода ближайших соседей для поиска ближайших соседей к целевому объекту.
    # Указываем косинусную метрику. Чем она меньше, тем ближе объекты в векторном пространстве(более похожи изображения)
    # В качестве алгоритма указан brute-force search (попарно сравнивает вектора)
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    vecs = vectorize_all(files, model, n_dims=4096)
    knn.fit(vecs)

    fname = 'data/target_image.jpg'
    vec = vectorize(fname, model)

    # Метод kneighbors() возращает значения dist, которые определяют, насколько "близки" картинки с target
    # изображением (количество возращаемых соседей = количеству изображений)
    dist, indices = knn.kneighbors(vec.reshape(1, -1), n_neighbors=len(files))
    dist, indices = dist.flatten(), indices.flatten()
    # Получаем список из пар "имя файла-значение метрики" в порядке возрастания её значения. Чем она меньше,
    # тем более похожи изображения.
    similar_images = [(files[indices[i]], dist[i]) for i in range(len(indices))]

    print('Картинка'.ljust(65), end='')
    print('Метрика')
    for image, dist in similar_images:
        print(f'{image:60} {dist}')
