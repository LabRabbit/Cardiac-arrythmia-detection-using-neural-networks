# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import json
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("testing.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:279]
Y = dataset[:,279]
# create model
model = Sequential()
model.add(Dense(300, input_dim=279, kernel_initializer='uniform', activation='relu'))
model.add(Dense(200, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model

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
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#predictions = model.predict(X)

# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
#scores = model.evaluate(X, Y)
##print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

