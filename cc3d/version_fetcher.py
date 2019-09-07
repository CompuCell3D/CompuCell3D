import sys, os
path = sys.argv[1]
sys.path.append(path)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
import cc3d
enablePrint()

print(cc3d.__version__)