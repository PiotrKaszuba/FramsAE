import glob
import shutil
import os
from pathlib import Path


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

home = str(Path.home())
datafolder = os.path.join(home, 'dataFolder2')
for fname in glob.glob(datafolder+'/**/logbook_*', recursive=True):
    print(fname)
    dir, f = os.path.split(fname)
    dir, name = os.path.split(dir)
    target_f = os.path.join(datafolder, 'logbooks', name)
    mkdir_p(target_f)

    shutil.copyfile(fname, os.path.join(target_f, f))