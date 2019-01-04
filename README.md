# Malaria Cell Recognition Model
A Convolutional Neural Net designed to recognize cells with Malaria. I used Transfer Learning from the Resnet50 model,
and trained the last layer on a medium sized dataset. 

**So far, the current model is 93% accurate.**

# Latest Run
```
725s 526ms/step - loss: 0.2549 - acc: 0.9004 - val_loss: 0.1962 - val_acc: 0.9303
```

# Dataset
The train dataset contain 11024 Parasitized cell images and 11024 Uninfected cell images.

The Test dataset contains 2756 Parasitized cell images and 2756 Uninfected cell images.

# How to use
Download the model json from ```model/malaria_recognizer.json``` and the weights files from  ```model/malaria_recognizer.h5```

This is a Keras Convulutional Neural Net model using Stochastic Gradient Descent and categorical crossentropy as the 
loss function. You can import it into any python project as such:
```
# Load the model from json file
json_file = open(path_to_model, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights into new model
model.load_weights(path_to_weights)

# Compile the model
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
```