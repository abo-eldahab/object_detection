import pickle as p
import matplotlib.pyplot as plt
import numpy as np
from cv2 import resize

def prediction(image , model=''):
    if model == 'RandomRorest':
        loaded_model = p.load(open('RandomForest_model_dataset2_pickle', 'rb'))
        fruit_list = p.load(open('fruit_names_RandomForest_dataset2', 'rb'))
    elif model == 'DecisionTree':
        loaded_model = p.load(open('DecisionTree_model_dataset2_pickle', 'rb'))
        fruit_list = p.load(open('fruit_names_DecisionTree_dataset2', 'rb'))

    img =  plt.imread(image)
    img = resize(img, (150,150))
    img = img.ravel()
    np.array(img)
    f = loaded_model.predict([img])
    s = loaded_model.predict_proba([img])
    predict_proba = dict(zip(fruit_list, [t for t in s[0]]))
    return fruit_list[f[0]] , predict_proba