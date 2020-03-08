import re

dict = 'XqQrRlLcCfFmM()'

def parseGenFile(path):
    f = open(path, 'r')
    text = f.read()
    genos = re.findall(r":\~\n[^\n]+\~", text, re.U)
    for i in range( len(genos)):
        genos[i] = genos[i][3:-1]
    return genos

def limitDict(genos, dict):
    for i in range(len(genos)):
        genos[i] = re.sub('[^'+dict+']', '', genos[i])
    return genos


def testGenos():
    genos = parseGenFile("encoding_f1_best.gen")

    genos = limitDict(genos, dict)
    return genos

if __name__ == "__main__":
    testGenos()