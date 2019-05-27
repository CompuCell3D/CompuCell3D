import sys
from os import environ

envVarName = sys.argv[1]

envVarStr = environ[envVarName]

envVarList = envVarStr.split(';')

# removing duplicates from the list while keeping original order
from collections import OrderedDict

envVarList = list(OrderedDict.fromkeys(envVarList))

envVarStr1 = ';'.join(envVarList)

command_str = 'SET PATH=' + envVarStr1 + '\n'

print(command_str)
