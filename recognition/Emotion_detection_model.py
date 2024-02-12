import tensorflow as tf #open source library primarily used for large numerical Computation
import numpy as np #for working with arrays
import pandas as pd #for data analysis
import cv2 #opencv lib: used for opening the camera
import keras # Used for creating deep Learning models like ANN, CNN
# from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
# from tensorflow.keras.applications import ResNet50 #CNN model That is 50 layers deep and is trained on a million images of 1000 categories.


df1 = pd.read_csv("../input/fer2013/fer2013.csv") #reading the dataset

# 0 - Angry
# 1 - Disgust
# 2 - Fear
# 3 - Happy
# 4 - Sad
# 5 - Surprise
# 6 - Neutral

print(df1.emotion.value_counts())  #counting total no of values in it
print(df1.head())


# Preprocessing
x_train=[]
x_test=[]
y_train=[]
y_test=[]
for i,row in df1.iterrows(): # iterating rows in dataset
    k=row['pixels'].split(" ")   # the "pixels" column contains the numerical representation of the grayscale images # splits a string into a list.
    if(row['Usage']=='Training'):
        x_train.append(np.array(k))  #Adding all the pixel list of grayscale image in that list
        y_train.append(row['emotion']) # The "emotion" column contains integers representing the emotion labels of the facial expression images. 
        #Each row in the "emotion" column corresponds to an image in the dataset and contains an integer representing the emotion label of that image.
    elif(row['Usage']=='PublicTest'):
        x_test.append(np.array(k)) #Adding all the pixel list of grayscale image in that list
        y_test.append(row['emotion']) # The "emotion" column contains integers representing the emotion labels of the facial expression images. 
        #Each row in the "emotion" column corresponds to an image in the dataset and contains an integer representing the emotion label of that image.

x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

x_train=x_train.reshape(x_train.shape[0],48,48) #reshaping the list into image of 48 X 48 pixels
x_test=x_test.reshape(x_test.shape[0],48,48) #reshaping the list into image of 48 X 48 pixels
y_train=tf.keras.utils.to_categorical(y_train,num_classes=7)  # returns a one-hot encoded matrix representation of the labels.  useful for training machine learning models to classify multi-class problems. 
y_test=tf.keras.utils.to_categorical(y_test,num_classes=7) # returns a one-hot encoded matrix representation of the labels.  useful for training machine learning models to classify multi-class problems.


# Showcasing some grayscale images
import matplotlib.pyplot as plt
for i in range(10):
  image=x_test[i].reshape((48,48)) 
  image=image.astype('float32')
  print(image.shape)
  plt.imshow(image,cmap=plt.cm.gray)
  plt.show()

#data augmentation
x_train=x_train.reshape((x_train.shape[0],48,48,1)) #converts the 3d dims image into 4 , where last column is used for color channel , 1 : only one color
x_test=x_test.reshape((x_test.shape[0],48,48,1)) #converts the 3d dims image into 4 , where last column is used for color channel , 1 : only one color

# ImageDataGenerator: Augment your images in real-time while your model is still training
# augmentation is a technique of applying different transformations to original images which results in multiple transformed copies of the same image
# ImageDataGenerator class ensures that the model receives new variations of the images at each epoch.

from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(rescale=1./255, #normalizing the pixel values between 0 and 1 from 0 to 255 values
                                   rotation_range=60, # specifies the range of random rotations to apply to the images, in degrees.
                                   shear_range=0.5, #specifies the range of random shearing to apply to the images.
                                   zoom_range=0.5, # the range of random zooming to apply to the images.
                                   width_shift_range=0.5, #specify the range of random horizontal and vertical shifts to apply to the images.
                                   height_shift_range=0.5,
                                   horizontal_flip=True, #specifies whether to randomly flip the images horizontally.
                                   fill_mode='nearest') # specifies the method used to fill in any missing pixels that may be created during the image transformations

validation_datagen = ImageDataGenerator(rescale=1./255) #normalizing the pixel values between 0 and 1

train_datagen.fit(x_train)
validation_datagen.fit(x_test)

print(x_train.shape)

#applying CNN model

model1=keras.models.Sequential()

# Block-1

#The 32 is the number of features we are extracting. 

model1.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                     kernel_initializer='he_normal',
                     activation="elu",  # decides whether to activate an neuron or not, Exponential Linear Unit where it smoothen's the negative wieghts
                     input_shape=(48,48,1))) # resultant shape of an input image
model1.add(keras.layers.BatchNormalization()) #normalizes the output of each layer by subtracting the mean and dividing by the standard deviation of the batch.

model1.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                     kernel_initializer='he_normal', 
                     activation="elu"))

model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.MaxPooling2D(pool_size=(2,2))) #max pooling reduces the dimensionality of images by reducing the number of pixels in the output from the previous convolutional layer
model1.add(keras.layers.Dropout(0.2)) # drops some neurons. # used for reducing the problem of overfitting

# Block-2
model1.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', 
                     kernel_initializer='he_normal',
                     activation="elu"))

model1.add(keras.layers.BatchNormalization())

model1.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',
                     kernel_initializer='he_normal', 
                     activation="elu"))

model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(keras.layers.Dropout(0.2))

# Block-3
model1.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', 
                     kernel_initializer='he_normal', 
                     activation="elu"))
model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', 
                     kernel_initializer='he_normal',
                     activation="elu"))

model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(keras.layers.Dropout(0.2))

# Block-4
model1.add(keras.layers.Flatten())
model1.add(keras.layers.Dense(64, activation="elu", kernel_initializer='he_normal'))
model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.Dropout(0.5))

# Block-5
model1.add(keras.layers.Dense(7, activation="softmax", kernel_initializer='he_normal'))

print(model1.summary()) #final summary of our model

#Model Plot
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils

keras.utils.plot_model(model1, to_file='model.png', show_layer_names=True) #displaying the process in an image

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

#intializing callbacks
early_stopping=keras.callbacks.EarlyStopping(patience=15,restore_best_weights=True) # callback that stops the training of the model if there is no improvement in the validation loss for a certain number of epochs    
filepath="weights/weights.best.hdf5"

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') # callback that stops the training of the model if there is no improvement in the validation loss for a certain number of epochs 

# The verbose parameter is set to 1, which means that a message will be printed to the console when the weights are saved. 
# The save_best_only parameter is set to True, which means that only the weights from the epoch with the highest validation accuracy will be saved 
# to the file. The mode parameter is set to 'max', which means that the model will be saved based on the maximum value of the monitored quantity (validation accuracy in this case).



#optimization algorithm : optimizies weights and reduces losses
model1.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


#fitting the model and running it for 50 epochs (epochs means iterations)
model1.fit(x_train,y_train,
           batch_size=64,
           epochs=50,
           validation_data=(x_test,y_test),
           verbose=1,callbacks=[early_stopping])

print(model1.evaluate(x_test,y_test)) #evaluating the model

#saving the model
fer_json = model1.to_json()  
with open("fer.json", "w") as json_file:  
    json_file.write(fer_json)  
model1.save_weights("fer.h5")


