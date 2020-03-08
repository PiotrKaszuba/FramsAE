import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from ParseGenFile import dict, testGenos

def encodeGenos(genos, encoders):
    for i in range(len(genos)):
        genos[i] = oneHotEncode(genos[i], encoders)
    return genos


def prepareEncoders(dict):
    dict = list(dict)
    values = np.array(dict)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, onehot_encoder

def oneHotEncode(geno, encoders):
    data = list(geno)
    values = np.array(data)
    print(values)


    integer_encoded = encoders[0].transform(values)
    print(integer_encoded)


    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = encoders[1].transform(integer_encoded)
    print(onehot_encoded)
    return onehot_encoded

def testEncode():
    genos = testGenos()
    encoders = prepareEncoders(dict)
    genos_encoded = encodeGenos(genos, encoders)
    return genos_encoded

if __name__ == "__main__":
    testEncode()