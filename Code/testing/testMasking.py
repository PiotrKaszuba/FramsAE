from keras.layers import *
from keras import Model
import numpy as np

x=np.zeros(shape=(10,2))
x[0,0] = 1
x[1,0] = 1
x[2,1]=1
inp = Input(shape=(2,))
embedLayer = Embedding(2, 3, mask_zero=False, trainable=False)

embed = embedLayer(inp)
print(embedLayer.get_weights())
embedLayer.set_weights(np.zeros_like(embedLayer.get_weights()))
print(embedLayer.get_weights())
lstm = LSTM(1, return_sequences=True)(embed)
model = Model(inputs=inp, outputs=lstm)

model.compile(optimizer='adam', loss='binary_crossentropy')
y=model.predict(x)

print(y)
print(embedLayer.get_weights())