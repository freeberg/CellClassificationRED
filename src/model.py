import sys
import tensorflow as tf

from tensorflow.keras import backend as K
K.image_data_format()


import os
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import Augmentor
from red_gen import gen_red_generators
from model_gen import gen_model, add_dense_layers
#import torchvision

train_dir = "data/SUS_318_10/train"
val_dir = "data/SUS_318_10/validation"
test_dir = "data/SUS_318_10/test"
num_classes = 4
input_shape = (318, 318)
batch_size = 40

red_prob = 0.0
red_grid = [16,16]
red_magnitude = 8

# Generate image generators with random elastic distortions (red)
train_gen, val_gen, test_gen = gen_red_generators(train_dir, val_dir, test_dir, 
                                 input_size=input_shape, batch_size=batch_size,
                                 class_mode='categorical', red_prob=red_prob,
                                 red_grid=red_grid, red_magnitude=red_magnitude)

nbr_layers = 4
nbr_kernels = [16, 32, 64, 128]
kernel_size=[[3,3], [3,3], [3,3], [3,3]]
strides=[[1,1], [1,1], [1,1], [1,1]]
nbr_dense_layers = 2
hidden_nodes = [18, num_classes]
# Generate model 
model = gen_model(nbr_layers=nbr_layers, nbr_kernels=nbr_kernels, kernel_size=kernel_size, 
                  strides=strides, input_shape=train_gen.image_shape)
# model.add(Dropout(0.4))
model = add_dense_layers(model, nbr_layers=nbr_dense_layers, nbr_hidden_nodes=hidden_nodes,
                         final_actfunc='softmax')
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=5e-4), metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')

history = model.fit_generator(train_gen, steps_per_epoch=int(109003/batch_size), verbose=1, epochs=25, validation_data=val_gen, callbacks=[early_stop])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('model_cells%d.h5' % 264)

extra, test_acc = model.evaluate(test_gen, verbose=0)

print(extra)
print(test_acc)

#0.9325703444980806
#0.81204593