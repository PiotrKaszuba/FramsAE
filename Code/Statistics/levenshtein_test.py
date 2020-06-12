from Levenshtein import distance as levenshtein_distance
from Code.Preparation.ParseGenFile import testGenos, representation, howMany
import numpy as np


inds = np.triu_indices(howMany, 1)

matrix = np.zeros((howMany,howMany), dtype='float64')



genos = testGenos(representation+"TestDissim.gen")

import pandas as pd

df = pd.read_csv("outputfilename"+representation+"-2",sep='\t',skiprows=(0,1,2,3),header=None).drop(columns=[0,1]).values


for i,j in zip(inds[0], inds[1]):
    matrix[i,j] = levenshtein_distance(genos[i], genos[j])

valsL= matrix[inds]
valsFit = df[inds]

valsRand = np.random.rand(np.shape(valsFit)[0])/2.5 - 0.15

corr = np.corrcoef(valsL, valsFit)
corr1 = np.corrcoef(valsFit, valsRand)
corr2 = np.corrcoef(valsRand, valsFit)
print("Correlation:")
print(corr)





