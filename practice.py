# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pickle
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians data-clean-imputed-clean-imputedset
data = numpy.loadtxt("data_clean_imputed.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = data[:,0:279]
Y = data[:,279]
# create model
model = Sequential()
model.add(Dense(300, input_dim=279, init='uniform', activation='sigmoid'))
model.add(Dense(200, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=40)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

