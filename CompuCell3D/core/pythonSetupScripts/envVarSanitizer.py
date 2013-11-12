import sys
from os import environ

sys.path.append(environ["PYTHON_MODULE_PATH"]) # needed on Mac with Python 2.5

envVarName=sys.argv[1]


envVarStr=environ[envVarName]
# print 'envVarStr=',envVarStr

envVarList=envVarStr.split(';')
# print 'envVarList=',envVarList

#removing duplicates from the list while keeping original order
from collections import OrderedDict
envVarList=list(OrderedDict.fromkeys(envVarList))

# print 'envVarList 1=',envVarList

envVarStr1=';'.join(envVarList)

# print 'envVarStr1=',envVarStr1

# environ[envVarName]=envVarStr1

# print 'PATH IN PYTHON=',environ[envVarName]

print envVarStr1

