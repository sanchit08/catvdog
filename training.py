import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import training
from tensorflow.keras.constraints import max_norm



x,y = training.create_training_data()   #load the data created using load
x = x/255.0    #normalize the image data by scaling from 0-255 to 0-1

#split the dataset into training and testing dataset
xTrain, xTest, yTrain, yTest= train_test_split(x,y, test_size=0.01, random_state=0)


model =  Sequential()
model.add(Conv2D(256, (3,3), input_shape=(100,100,1), padding='same', kernel_constraint=max_norm(3)))   #1st hiddden layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3),padding='same', kernel_constraint=max_norm(3)))    #2nd hidden layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 

model.add(Flatten())   #convert 3d output vector to 1d vector

model.add(Dense(64, activation='relu'))        #final output layer
model.add(Dense(1, activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy',optimizer = 'adam', metrics=['accuracy'])


#train the data 
model.fit(xTrain, yTrain, epochs=10, batch_size=25, verbose=1)
	
#evaluate the model on the test dataset
_,accuracy= model.evaluate(xTest,yTest)
print('Accuracy: ')
print(accuracy*100)  
 
#save the created model and weights to a h5 file
model.save('cvd.h5')




