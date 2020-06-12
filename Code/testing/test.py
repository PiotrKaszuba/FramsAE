import numpy as np



vectors_rem = np.random.rand(2,4)

#https://en.wikipedia.org/wiki/Orthogonal_complement
def get_orthogonal_complement_full_matrix(vectors):

    def zeroOutDimension(vectorZeroed, vectorZeroing, zeroingInd):
        vectorZeroed -= vectorZeroing * vectorZeroed[zeroingInd] / vectorZeroing[zeroingInd]

    def normalizeDimension(vectorNormalized, normalizedInd):
        vectorNormalized /= vectorNormalized[normalizedInd]

    for zeroingInd, vectorZeroing in enumerate(vectors):
        for zeroedInd, vectorZeroed in enumerate(vectors):
            if zeroingInd!=zeroedInd:
                zeroOutDimension(vectorZeroed, vectorZeroing, zeroingInd)

    for normalizedInd, vectorNormalized in enumerate(vectors):
        normalizeDimension(vectorNormalized, normalizedInd)

    matrix = np.identity(np.shape(vectors)[1])
    matrix[:, :np.shape(vectors)[0]] = np.transpose(-vectors)
    matrix[:np.shape(vectors)[0]] = vectors


    return matrix

def transform_vector_space(vector_space, transformation_matrix):
    return np.linalg.inv(transformation_matrix) @ vector_space

print(get_orthogonal_complement_full_matrix(vectors_rem))



x=0