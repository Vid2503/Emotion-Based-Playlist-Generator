from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPool2D,Flatten,Dropout
import os
test='data/test/'
train='data/train/'
#data augmentation
traingen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest"
)
testgen=ImageDataGenerator(
    rescale=1./255
)
newtrain=traingen.flow_from_directory(
    train,
    target_size=(48,48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=True
)
newtest=testgen.flow_from_directory(
    test,
    target_size=(48,48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=True
)
img,lbl=newtrain.__next__()
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.1))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.1))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
epochs=30
history=model.fit(newtrain,
                steps_per_epoch=28709//32,

                epochs=epochs,
                validation_data=newtest,
                validation_steps=7178//32)
model.save('model_file.h5')
