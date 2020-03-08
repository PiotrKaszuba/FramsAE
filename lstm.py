from keras.layers import *
from keras.models import Model

inpE = Input((10,5)) #here, you don't define the batch size
outE = LSTM(units = 20, return_sequences=False)(inpE)

encoder = Model(inpE,outE)
