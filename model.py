# import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, metrics
import matplotlib.pyplot as plot
import numpy as np
import math
import scipy.ndimage
import glob
import imageio.v3 as iio
import rasterio
from scipy.ndimage import zoom
from image_operations import *

image_path_plumes = "/coursedata/plumes/*.png"
image_path_nplumes = "/coursedata/n-plumes/*.png"

#create a tensor
n_plumes = len(glob.glob(image_path_plumes))
n_nplumes = len(glob.glob(image_path_nplumes))
print(n_plumes)
print(n_nplumes)
#kxnxnxl tensor where k=number n=dimension l=label
Aplumes = np.zeros((n_plumes, 64, 64))
i = 0
for f in glob.glob(image_path_plumes):
    im = image_loader(f, 64, 64)
    Aplumes[i] = im
    i = i+1
Anplumes = np.zeros((n_nplumes, 64, 64))
j = 0
for f in glob.glob(image_path_nplumes):
    im = image_loader(f, 64, 64)
    Anplumes[j] = im
    j = j+1


#label plume images as 1 and non plumes as 0
original_imagedata = np.concatenate((Aplumes, Anplumes),0)
original_labels = np.concatenate((np.ones((n_plumes)), np.zeros((n_nplumes))),0)

#augment the data with invariant transformations
imagedata = invariant_transformation(original_imagedata)
labels = original_labels
for i in range(0,7):
    t_imagedata = invariant_transformation(original_imagedata)
    imagedata = np.concatenate((imagedata,t_imagedata),0)
    labels = np.concatenate((labels,original_labels),0)

#shuffle
print(np.shape(imagedata))
print(np.shape(labels))
N = len(imagedata)
p = np.random.permutation(N)
imagedata = imagedata[p]
labels = labels[p]
#splits
bigger_chunk = math.ceil(N*0.7)
smaller_chunk = math.ceil(N*0.15)

x_train = imagedata[0:bigger_chunk]
y_train = labels[0:bigger_chunk]
x_test = imagedata[bigger_chunk:bigger_chunk+smaller_chunk]
y_test = labels[bigger_chunk:bigger_chunk+smaller_chunk]
x_valid = imagedata[bigger_chunk+smaller_chunk:]
y_valid = labels[bigger_chunk+smaller_chunk:]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,1),padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(layers.Flatten())
#model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', metrics.Precision(), metrics.Recall()])

history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_valid, y_valid))

plot.plot(history.history['accuracy'], label='accuracy')
plot.plot(history.history['val_accuracy'], label = 'val_accuracy')
#plot.plot(history.history['precision'], label = 'precision')
#plot.plot(history.history['recall'], label = 'recall')
plot.xlabel('Epoch')
plot.ylabel('Performance')
plot.ylim([0.5, 1])
plot.legend(loc='lower right')

test_loss, test_acc, test_pred, test_recall = model.evaluate(x_test,  y_test, verbose=2)

y_pred = model.predict(x_test)

N = len(x_test)
k = np.random.randint(0, N, size=4)
fig, axes = plot.subplots(2,2, figsize=(10,10))
axes[0,0].imshow(x_test[k[0]], cmap='viridis')
axes[0,0].set_title("Prediction: " + str(y_pred[k[0]]) + " True label: " + str(y_test[k[0]]))
axes[0,1].imshow(x_test[k[1]], cmap='viridis')
axes[0,1].set_title('Prediction: ' + str(y_pred[k[1]]) + ' True label: ' + str(y_test[k[1]]))
axes[1,0].imshow(x_test[k[2]], cmap='viridis')
axes[1,0].set_title('Prediction: ' + str(y_pred[k[2]]) + ' True label: ' + str(y_test[k[2]]))                                         
axes[1,1].imshow(x_test[k[3]], cmap='viridis')
axes[1,1].set_title('Prediction: ' + str(y_pred[k[3]]) + ' True label: ' + str(y_test[k[3]]))

