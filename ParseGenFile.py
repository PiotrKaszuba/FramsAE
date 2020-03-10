import re

dict = 'XqQrRlLcCfFmM()iI,ST'

def parseGenFile(path):
    f = open(path, 'r')
    text = f.read()
    genos = re.findall(r"genotype:[^\n]+\n", text, re.U)
    #genos = re.findall(r":\~\n[^\n]+\~", text, re.U)
    for i in range( len(genos)):
        #genos[i] = genos[i][3:-1]
        genos[i] = genos[i][9:-1]
        genos[i] = 'S' + genos[i] + 'T'
    return genos

def limitDict(genos, dict):
    for i in range(len(genos)):
        genos[i] = re.sub('[^'+dict+']', '', genos[i])
    return genos


def testGenos():
    genos = parseGenFile("customGens.gen")

    genos = limitDict(genos, dict)
    return genos

if __name__ == "__main__":
    testGenos()