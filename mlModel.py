
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

dataset=load_digits()

data=dataset.data
images=dataset.images
target=dataset.target

#print('data shape:',data.shape)
#print('images shape:',images.shape)
#print('target shape:',target.shape)

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.2)

algorithm=KNeighborsClassifier(n_neighbors=3)
algorithm.fit(train_data,train_target)

predict_targets=algorithm.predict(test_data)

#accuracy
#print("predict targets:",predict_targets)
#print("actual targets:",test_target)

acc=accuracy_score(test_target,predict_targets)
print(f"Model Accuracy: {acc * 100:.2f}%")

joblib.dump(algorithm,'KNN-digit.sav')