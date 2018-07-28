
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2, numpy as np, os.path
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf
import theano as th
from keras import backend as K  
K.set_image_dim_ordering('tf')
K.set_epsilon(1e-07)
K.set_image_data_format('channels_first')
K.set_floatx('float32')



class Model(object):

    FILE_PATH = './faces6.h5'

    def __init__(self):
        self.model = None

    def train(self, batch_size, classes,epochs):
        self.batch_size=batch_size
        self.epochs=epochs
        
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150), padding = 'same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (3, 3), padding = 'same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(64, (3, 3), padding = 'same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model.summary()
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'data/train',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

        
        validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical')

        steps_per_epoch=5000 // self.batch_size
        validation_steps=400 // self.batch_size
        
        
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)
        
  

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    def predict(self, image):
        
        image=cv2.resize(image,(150,150),interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image=img_to_array(image)
        #print(image.shape)
        image = image.reshape((1,) + image.shape)
        #print(image.shape)
        image = image.astype('float32')
        image /= 255
        result = self.model.predict_proba(image)
        #print(result)
        result = self.model.predict_classes(image)

        return result
        


if __name__ == '__main__':
    model = Model()
    fname=model.FILE_PATH
    if os.path.isfile(fname) is True: 
        #model.load()
        print("")
    else :
        model.train(batch_size=32, classes=6,epochs=15)
        model.save()
        #model.load()
        

    


# In[2]:

from random import randint

nameList = ['Hermione', 'Hermione', 'Ron', 'Malfoy', 'Malfoy', 'Malfoy', 'Chang', 'Harry','Snape','Ron',
            'Chang', 'Harry', 'Chang', 'Snape', 'Change', 'Snape', 'Harry', 
            'Malfoy','Ron','Hermione','Ron']
responseList = ['Welcome back, Harry!','Welcome back, Hermione!','Welcome back, Ron!','You don\'t belong here, Chang!','You don\'t belong here, Malfoy!','You don\'t belong here, Snape!']

model = Model()

count=str(randint(1,22))
show = "./data/test/"+count+".png"
image = cv2.imread(show)

image = image[:,:,::-1]
model.load()
model.predict(image)
print("image: ", show)
print("Real identity: ", nameList[int(count)-1])
print("Fat Lady: \"{}\"".format(responseList[int(model.predict(image))]))


