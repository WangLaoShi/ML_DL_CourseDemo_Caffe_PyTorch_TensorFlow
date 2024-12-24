# Create your first MLP in Keras
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from keras.models import Sequential,model_from_json,model_from_yaml
from keras.layers import Dense
import numpy
from keras.models import save_model, load_model
# ImportError: You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model to work.
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # https://keras.io/zh/layers/core/#dense
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
# history = model.fit(X, Y, epochs=150, batch_size=10)
# Fit the model
# history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
history = model.fit(X, Y, validation_split=0.33, epochs=1500, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
"""
From the plot of the accuracy, you can see that the model could 
probably be trained a little more as the trend for accuracy on 
both datasets is still rising for the last few epochs. You can 
also see that the model has not yet over-learned the training 
dataset, showing comparable skill on both datasets.
"""
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
"""
From the plot of the loss, you can see that the model has comparable performance on both 
train and validation datasets (labeled test). If these parallel plots start to depart 
consistently, it might be a sign to stop training at an earlier epoch.
"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

from ann_visualizer.visualize import ann_viz;

"""
ann_viz(model, view=True, filename=”network.gv”, title=”MyNeural Network”)
model – Your Keras sequential model
view – If set to true, it opens the graph preview after the command has been executed
filename – Where to save the graph. (it’s saved in a ‘.gv’ file format)
title – The title for the visualized ANN
"""

ann_viz(model, title="My first neural network")
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))

# pip install h5py

# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

# pip install PyYAML

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
print("Saved model to disk")

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# Save Model Weights and Architecture Together
# equivalent to: model.save("model.h5")
save_model(model, "model.h5")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

# load and evaluate a saved model
from numpy import loadtxt

# load model
model = load_model('model.h5')
# summarize model.
model.summary()
# load dataset
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# evaluate the model
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))