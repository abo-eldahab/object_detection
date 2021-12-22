import services as service
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import confusion_matrix


images = service.extract_dataset(path_dataset="dataset2/")

df = service.prepare_dataframe(images=images)

fruit_names = sorted(df.fruit.unique())

service.dump_data_to_pickle(data=fruit_names , filename="fruit_names_DecisionTree_dataset2")

mapper_fruit_names = dict(zip(fruit_names, [t for t in range(len(fruit_names))]))
df["label"] = df["fruit"].map(mapper_fruit_names)

service.draw_chart(df=df)

model = DecisionTreeClassifier()
start_time = time.time()
X , Y = service.load_img(df=df)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
model.fit(X_train, y_train)
time_model = time.time() - start_time

print(f"Time to train the model: {int(time_model)} seconds")

y_pred = model.predict(X_test)
    
cm = confusion_matrix(y_test, y_pred)
print(cm)

score=model.score(X_test,y_test)
print(score)

service.dump_data_to_pickle(data=model , filename="DecisionTree_model_dataset2_pickle")