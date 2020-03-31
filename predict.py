import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model



#load the model
x = []
model = load_model('cvd.h5')
path = '/Predict'
q = os.listdir(path)


#load the images required to be predicted

for i in range(len(q)):
	img = cv2.imread(os.path.join(path,q[i]), cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (100,100))
	x.append(img)
x = np.array(x).reshape(-1,100,100,1)    #convert the loaded images into a numpy array

#normalize the data
x = x/255.0

#predict the output of required images
prediction = model.predict_classes(x)

#print the data
for i in range(len(x)):
	if prediction[i] == 0:
		print(q[i] +' '+ 'is a dog')
	else:
		print(q[i] +' '+ 'is a cat')
