from keras.layers import *
from keras.models import Model, Sequential
from encodeGenos import testEncode
from keras.preprocessing.sequence import pad_sequences

sequences = testEncode()

sequence = pad_sequences(sequences, padding='post', dtype='float32')
max_len = sequence.shape[1]
features = sequence.shape[2]

inp = Input(shape=(max_len, features))
encoder = LSTM(8, activation='relu')(inp)

# define reconstruct decoder
decoder1 = RepeatVector(max_len)(encoder)
decoder1 = LSTM(8, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(features))(decoder1)

model = Model(inputs=inp, outputs=decoder1)


model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(sequence, sequence,
                      epochs=400,
                      verbose=2,
                      validation_split=0.1,
                      shuffle=True)