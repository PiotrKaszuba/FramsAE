import re
import numpy as np
import os
import framsreader as fr
def getDictFromGenos(genos):
    dict = set()
    for i in range(len(genos)):
        dict.update(genos[i])
    remove = set('/*49*/ ')
    dict = dict.difference(remove)
    dict = ''.join(dict)
    print(dict)
    print(len(dict))


def parseFitness(path):
    f = open(path, 'r')
    text = f.read()
    fitness = re.findall(r"vertpos:[^\n]+\n", text, re.U)
    for i in range(len(fitness)):
        fitness[i] = float(fitness[i][8:-1])
    return fitness

def parseGenFile(path, representation, dict = None, representation_match_part=r''):
    #path = 'f4-len-up-tp-100.gen'
    f = open(path, 'r')
    text = f.read()
    # if dict is not None:
    #     text = re.sub(r"(genotype:" + representation_match_part +r")([^\n]+\n)", lambda x: x.group(1) + re.sub('[^\n' + dict + ']', '', x.group(2)), text)
    #     f2 = open(path+"_copy", 'w')
    #     f2.write(text)
    genos = re.findall(r"genotype:[^\n]+\n", text, re.U)
    #genos = re.findall(r":\~\n[^\n]+\~", text, re.U)
    additional_symbols = 0
    if representation != 'f1':
        additional_symbols += 5
    for i in range( len(genos)):
        #genos[i] = genos[i][3:-1]
        genos[i] = genos[i][9+additional_symbols:-1]
        genos[i] = 'S' + genos[i] + 'T'

    # genos = fr.load(path)
    #
    # if representation != 'f1':
    #     parse_f = lambda x: 'S'+x['genotype'][5:]+'T'
    # else:
    #     parse_f = lambda x: 'S'+x['genotype']+'T'
    #
    # genos = [parse_f(gen) for gen in genos]

    return genos

def limitDict(genos, dict):
    for i in range(len(genos)):
        genos[i] = re.sub('[^'+dict+']', '', genos[i])
    return genos



def testGenos(config, print_some_genos = False):
    nameSuff = config['representation'] +"_" + config['long_genos'] + ".gen"
    nameGens = os.path.join(config['load_dir'], "customGens" + nameSuff)
    genos = parseGenFile(nameGens, config['representation'], dict=config['dict'], representation_match_part=config['representation_match_part'])

    #nameFit = os.path.join(config['load_dir'], 'ocenione' + nameSuff)
    #fitness = parseFitness(nameFit)

    genos = limitDict(genos, config['dict'])



    if print_some_genos:
        leng = 0
        lens = []
        for gen in genos:
            lens.append(len(gen))

            leng += len(gen)
        print(np.max(lens))
        # print(np.max(fitness))
        # print(np.min(fitness))
        # print(np.mean(fitness))
        # print(np.std(fitness))
        # print(np.where(np.array(fitness) > 0, 1, 0).sum())
        print(leng/len(genos))
        for i in range(5):
            print(genos[i])

    return genos

# if __name__ == "__main__":
#     from Code.Preparation.configuration import get_config
#     config = get_config('', 'f4', 'short', 32, False, False, '', '')
#     testGenos(config, True)