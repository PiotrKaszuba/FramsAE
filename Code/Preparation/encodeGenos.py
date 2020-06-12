import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from Code.Preparation.ParseGenFile import testGenos




def encodeGenos(genos, encoders, oneHot):
    return [oneHotEncode(genos[i], encoders, oneHot) for i in range(len(genos))]



def prepareEncoders(dict):
    dict = list(dict)
    values = np.array(dict)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)

    return label_encoder, onehot_encoder


def inverse(onehot_genos, encoders, oneHot):
    len = np.shape(onehot_genos)[0]
    genos = [inverseEncode(onehot_genos[i], encoders, oneHot) for i in range(len)]
    return genos

def inverseEncode(onehot_encoded, encoders, oneHot=True):
    if oneHot:
        onehot_encoded = encoders[1].inverse_transform(onehot_encoded)
        onehot_encoded = np.reshape(onehot_encoded, (np.shape(onehot_encoded)[0],))
    #print(integer_encoded)
    geno = encoders[0].inverse_transform(onehot_encoded)
    #print(geno)
    return geno

def oneHotEncode(geno, encoders, oneHot=True):
    data = list(geno)
    values = np.array(data)
    #print(values)


    integer_encoded = encoders[0].transform(values)
    #print(integer_encoded)

    if oneHot:
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        integer_encoded = encoders[1].transform(integer_encoded)
    #print(onehot_encoded)
    return integer_encoded

def testEncode(oneHot):

    genos = testGenos()
    encoders = prepareEncoders(dict)
    genos_encoded = encodeGenos(genos, encoders, oneHot)
    genos_decoded = inverse(genos_encoded, encoders, oneHot)
    return genos_encoded, encoders

if __name__ == "__main__":
    testEncode(oneHot)