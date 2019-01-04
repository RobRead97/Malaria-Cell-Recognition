from tensorflow.python.keras.applications import ResNet50
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.resnet50 import preprocess_input
import time


num_classes = 2  # Parasitized & Uninfected
img_rows, img_cols = 112, 112
model_path = './model/'
resnet_weights_path = './ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(Dense(num_classes, activation='softmax'))

# Since the ResNet model is already trained, I won't train it.
model.layers[0].trainable = False

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# Reads the images from our training set. Batch size of 25 gives us 800 batches. I've already categorized the set.
train_generator = data_generator.flow_from_directory(
        './train',
        target_size=(img_rows, img_cols),
        batch_size=16,
        class_mode='categorical')

# Reads the images from our training set. Also already categorized.
validation_generator = data_generator.flow_from_directory(
        './test',
        target_size=(img_rows, img_cols),
        class_mode='categorical')

model.fit_generator(
        train_generator,
        validation_data=validation_generator)

# Save the model to disk for future use.
model_json = model.to_json()
with open(model_path + "malaria_recognizer.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(model_path + "malaria_recognizer.h5")
print("Saved model to disk")