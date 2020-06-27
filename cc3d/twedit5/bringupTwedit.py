from .windowsUtils import *
import sys

print("COMMAND LINE OPTIONS:", sys.argv)

showTweditWindowWithPIDInForeground(int(sys.argv[1]))

# showTweditWindowInForeground()        
