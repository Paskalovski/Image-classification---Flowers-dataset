import numpy as np
import matplotlib.pyplot as plt
from visual import visual

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers


## importing the dataset
(training_set, validation_set), dataset_info = tfds.load('tf_flowers',
    split=['train[:70%]', 'train[70%:]'], with_info=True, as_supervised=True)

training_set

## Data Preprocessing 
num_classes = dataset_info.features['label'].num_classes

num_training_examples = 0
num_validation_examples = 0

for example in training_set:
  num_training_examples += 1


for example in validation_set:
  num_validation_examples += 1
  
# Image reformating and creating batches 
for i, example in enumerate(training_set.take(5)):
  print('Image {} shape: {} label: {}'.format(i+1, example[0].shape, example[1]))
  
  
IMAGE_RES = 224

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE)

validation_batches = validation_set.map(format_image).batch(BATCH_SIZE)


## Defining and testing the first models 

# Transfer learning Using the MobileNet v2 model

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))

feature_extractor.trainable = False

model = tf.keras.Sequential([feature_extractor,layers.Dense(num_classes)])

model.summary()


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

EPOCHS = 6


history = model.fit(train_batches,epochs=EPOCHS,validation_data=validation_batches)


## Results visualization 
visual(history)


## Defining and testing the second models 

# Transfer learning Using the Inception v3 model 

IMAGE_RES = 299

URL = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3),trainable=False)

model_inception = tf.keras.Sequential([feature_extractor,tf.keras.layers.Dense(num_classes)])

model_inception.summary()


model_inception.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

EPOCHS = 6

history = model_inception.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)

## Results visualization 
visual(history)
