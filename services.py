import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import pickle as p
from cv2 import resize
import numpy as np
import seaborn as sns


def load_images_from_folder(folder,only_path = False, label = ""):
    if only_path == False:
        images = []
        for filename in os.listdir(folder):
            img = plt.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        return images
    else:
        path = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder,filename)
            if img_path is not None:
                path.append([label,img_path])
        return path

def extract_dataset(path_dataset):
    images = []
    for f in os.listdir(path_dataset):
        if "jpg" in os.listdir(path_dataset+f)[0] or "png" in os.listdir(path_dataset+f)[0] or "jpeg" in os.listdir(path_dataset+f)[0]:
            images += load_images_from_folder(path_dataset+f,True,label = f)
        else: 
            for d in os.listdir(path_dataset+f):
                images += load_images_from_folder(path_dataset+f+"/"+d,True,label = f)
    return images

def prepare_dataframe(images):
    df = pd.DataFrame(images, columns = ["fruit", "path"])
    df = shuffle(df, random_state = 0)
    df = df.reset_index(drop=True)

    return df

def dump_data_to_pickle(data , filename):
    with open(filename,'wb') as f:
        p.dump(data,f)
        f.close()

def draw_chart(df):
    vc = df["fruit"].value_counts()
    plt.figure(figsize=(10,10))
    sns.barplot(x = vc.index, y = vc, palette = "rocket")
    plt.title("Number of pictures of each category", fontsize = 15)
    plt.xticks(rotation=90)
    plt.show()
    plt.savefig('chart.png')

def load_img(df):
    img_paths = df["path"].values
    img_labels = df["label"].values
    X = []
    y = []
    for i,path in enumerate(img_paths):
        img =  plt.imread(path)
        img = resize(img, (150,150))
        img = img.ravel()
        label = img_labels[i]
        X.append(img)
        y.append(label)
    return np.array(X),np.array(y)