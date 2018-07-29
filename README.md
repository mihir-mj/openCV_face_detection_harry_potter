# Introduction
As Fat Lady is guarding the entrance to Gryffindor common room, 
this project is to help reduce her workload by recongizing if the person in the image belongs to Gryffindor. 

Tutorial reference: https://www.udemy.com/i-want-learn-cnn-ai-technology/
![Fat Lady Image](https://i.ytimg.com/vi/3-bXjK_5C8g/maxresdefault.jpg)

## Data Categories
* Harry Potter
* Ron Weasley
* Hermione Granger
* Draco Malfoy
* Cho Chang
* Snape

# Data Pre-proccessing
## Download images
Download images from google image, with extension `I'm a gentleman`.
## Generate more data
To have enough data for training, Keras `ImageDataGenerator` is used to `rescale`, `shear_range`, `zoom_range`, `horizontal_flip` 
 the original image and then use scale or slightly moving positions to generate more images.
```python
train_datagen = ImageDataGenerator( 
    rescale=1./255,
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True
  )
```
## Data Cleaning
Remove images that are too small.
```python
for root, dirs, files in os.walk(destinationPath, topdown=False): 
    for name in files: 
        size=os.path.getsize(os.path.join(root, name))
        if size < 3*1024 :
           os.remove(os.path.join(root, name)) 
```
# CNN Model
### Params
* input_shape
* Activation('relu'))
* Conv2D(32, (3, 3)
* MaxPooling2D
* Dropout()
* Flatten()
* Dense(512)
* Dense(8)

```python
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
        
        # Start training
        self.model.fit_generator( 
            generator=train_generator,
            steps_per_epoch=steps_per_epoch, 
            epochs=epochs, 
            validation_data=validation_generator, 
            validation_steps=validation_steps
          )
```

## Model Summary
```ruby
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_27 (Conv2D)           (None, 32, 150, 150)      896       
_________________________________________________________________
activation_38 (Activation)   (None, 32, 150, 150)      0         
_________________________________________________________________
max_pooling2d_26 (MaxPooling (None, 32, 75, 75)        0         
_________________________________________________________________
conv2d_28 (Conv2D)           (None, 32, 75, 75)        9248      
_________________________________________________________________
activation_39 (Activation)   (None, 32, 75, 75)        0         
_________________________________________________________________
max_pooling2d_27 (MaxPooling (None, 32, 37, 37)        0         
_________________________________________________________________
dropout_19 (Dropout)         (None, 32, 37, 37)        0         
_________________________________________________________________
conv2d_29 (Conv2D)           (None, 64, 37, 37)        18496     
_________________________________________________________________
activation_40 (Activation)   (None, 64, 37, 37)        0         
_________________________________________________________________
max_pooling2d_28 (MaxPooling (None, 64, 18, 18)        0         
_________________________________________________________________
dropout_20 (Dropout)         (None, 64, 18, 18)        0         
_________________________________________________________________
flatten_7 (Flatten)          (None, 20736)             0         
_________________________________________________________________
dense_13 (Dense)             (None, 64)                1327168   
_________________________________________________________________
activation_41 (Activation)   (None, 64)                0         
_________________________________________________________________
dropout_21 (Dropout)         (None, 64)                0         
_________________________________________________________________
dense_14 (Dense)             (None, 6)                 390       
_________________________________________________________________
activation_42 (Activation)   (None, 6)                 0         
=================================================================
Total params: 1,356,198
Trainable params: 1,356,198
Non-trainable params: 0
_________________________________________________________________
Found 6650 images belonging to 6 classes.
Found 1512 images belonging to 6 classes.
Epoch 1/15
156/156 [==============================] - 763s 5s/step - loss: 1.6539 - acc: 0.3141 - val_loss: 1.3788 - val_acc: 0.5443
Epoch 2/15
156/156 [==============================] - 754s 5s/step - loss: 1.3633 - acc: 0.4748 - val_loss: 1.1160 - val_acc: 0.6016
Epoch 3/15
156/156 [==============================] - 740s 5s/step - loss: 1.1772 - acc: 0.5559 - val_loss: 1.0727 - val_acc: 0.6458
Epoch 4/15
156/156 [==============================] - 741s 5s/step - loss: 1.0592 - acc: 0.6054 - val_loss: 0.9803 - val_acc: 0.6536
Epoch 5/15
156/156 [==============================] - 738s 5s/step - loss: 0.9576 - acc: 0.6401 - val_loss: 0.8422 - val_acc: 0.7161
Epoch 6/15
156/156 [==============================] - 736s 5s/step - loss: 0.8581 - acc: 0.6699 - val_loss: 0.8137 - val_acc: 0.7266
Epoch 7/15
156/156 [==============================] - 724s 5s/step - loss: 0.7918 - acc: 0.7052 - val_loss: 0.6995 - val_acc: 0.7630
Epoch 8/15
156/156 [==============================] - 722s 5s/step - loss: 0.7677 - acc: 0.7196 - val_loss: 0.7045 - val_acc: 0.7344
Epoch 9/15
156/156 [==============================] - 726s 5s/step - loss: 0.7183 - acc: 0.7276 - val_loss: 0.6202 - val_acc: 0.7891
Epoch 10/15
156/156 [==============================] - 723s 5s/step - loss: 0.6559 - acc: 0.7482 - val_loss: 0.6776 - val_acc: 0.7578
Epoch 11/15
156/156 [==============================] - 721s 5s/step - loss: 0.6594 - acc: 0.7488 - val_loss: 0.6444 - val_acc: 0.7760
Epoch 12/15
156/156 [==============================] - 721s 5s/step - loss: 0.6023 - acc: 0.7720 - val_loss: 0.5455 - val_acc: 0.7917
Epoch 13/15
156/156 [==============================] - 718s 5s/step - loss: 0.5855 - acc: 0.7770 - val_loss: 0.5824 - val_acc: 0.8125
Epoch 14/15
156/156 [==============================] - 727s 5s/step - loss: 0.5528 - acc: 0.7930 - val_loss: 0.5101 - val_acc: 0.8177
Epoch 15/15
156/156 [==============================] - 719s 5s/step - loss: 0.5494 - acc: 0.7935 - val_loss: 0.4831 - val_acc: 0.8385
Model Saved.
```
* Training set accuracy rate 79.35%
* Testing set accuracy rate 83.85%

# Predict
Randomly pick photo from testing directory and see if the simulator successfully recognize the person in the image.
## Interface
```python
Image: './data/test/1.png'
Real identity: Hermione
Fat Lady: Welcome back, Hermione!
```
## Code
```python
from random import randint

nameList = ['Hermione', 'Hermione', 'Ron', 'Malfoy', 'Malfoy', 'Malfoy', 'Chang', 'Harry','Snape','Ron',
            'Chang', 'Harry', 'Chang', 'Snape', 'Change', 'Snape', 'Harry', 
            'Malfoy','Ron','Hermione','Ron']
responseList = ['Welcome back, Harry!','Welcome back, Hermione!','Welcome back, Ron!',
                'You don\'t belong here, Chang!','You don\'t belong here, Malfoy!','You don\'t belong here, Snape!']

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
```
