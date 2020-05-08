import os
import glob

current = os.path.dirname(os.path.realpath(__file__))
folders = glob.glob(os.path.join(current, '*/*'))

for folder in folders:
    src = str(folder)
    dest = src.replace('MIX_', 'MIX_0:')
    dest = dest.replace('_KIDX', ':0_KIDX')
    os.rename(src, dest)


