# enabling with statement in python 2.5


import os
import sys
from . import XMLUtils
# from CC3DProject.enums import *
from .ParameterScanEnums import *
from collections import OrderedDict
import json
from pathlib import Path
from typing import Optional
import contextlib


def removeWhiteSpaces(_str):
    import re
    out_str = str(_str)
    pattern = re.compile(r'\s+')
    out_str = re.sub(pattern, '', out_str)
    return out_str


def extractListOfStrings(_valueStr):
    values = None
    pos = _valueStr.find(
        '"')  # searching for " if found this means we have composite strings with commas inside - hence we cannot split the string based on comma position
    if pos >= 0:
        values = _valueStr.split('",')
        for i in range(len(values)):
            val = values[i]
            val = val.replace('"', '')
            values[i] = val
    else:
        values = _valueStr.split('",')
    return values


def nextIterationCartProd(currentIteration, maxVals):
    '''Computes next iteration of cartesian product
    '''
    length = len(currentIteration)
    import copy

    nextIteration = copy.deepcopy(currentIteration)

    # print 'currentIteration=',currentIteration
    # determine the lowest index with currentIteration value less than max value
    idx = -1
    for i in range(length):
        if currentIteration[i] < maxVals[i]:
            idx = i
            break

    if idx == -1:
        return [0 for i in range(length)]  # rollover
        # raise StopIteration

    # increment currentIteration for idx and zero iterations for i<idx
    nextIteration[idx] += 1
    # print 'currentIteration=',currentIteration

    for i in range(idx):
        nextIteration[i] = 0

    return nextIteration


def getParameterScanCommandLineArgList(_simFileName):
    '''returns command line options for parameter scan WITHOUT actual run script. run script has to be fetched independently using getCC3DPlayerRunscriptPath or getCC3DPlayerRunscriptPath in SystemUtils
    '''
    import sys
    # # # from SystemUtils import getCC3DPlayerRunscriptPath

    # # # cc3dPath=getCC3DPlayerRunscriptPath()
    reminderArgs = sys.argv[1:-1]  # w skip first and last arguments
    # print 'reminderArgs=',reminderArgs

    # check if arg -i <simulation name> exist
    try:
        idx = reminderArgs.index('-i')
        # # # reminderArgs=reminderArgs[0:idx]+reminderArgs[idx+2:]
    except ValueError as e:
        # if -i <simulationName> does not exist we add it to command line
        reminderArgs = ['-i', _simFileName] + reminderArgs

    # cmdLineArgs=[cc3dPath]+reminderArgs
    cmdLineArgs = reminderArgs
    return cmdLineArgs


class XMLHandler:
    def __init__(self):
        self.lineNumber = 0
        self.accesspath = []
        self.currentElement = None
        self.indent = 0
        self.indentOffset = 4
        self.xmlString = ''
        self.lineToElem = {}
        self.lineToAccessPath = {}
        self.accessPathList = []
        self.cc3dXML2ObjConverter = None
        self.root_element = None

    def newline(self):
        self.xmlString += '\n'
        self.lineNumber += 1

    def outputXMLWithAccessPaths(self, _xmlFilePath):
        from . import XMLUtils
        import os
        self.cc3dXML2ObjConverter = XMLUtils.Xml2Obj()

        self.root_element = self.cc3dXML2ObjConverter.Parse(_xmlFilePath)
        xmlStr = self.writeXMLElement(self.root_element)
        # print 'xmlStr=',self.xmlString

    def writeXMLElement(self, _elem, _indent=0):
        self.indent = _indent
        from . import XMLUtils
        import copy
        spaces = ' ' * self.indent

        self.xmlString += spaces + '<' + _elem.name
        currentElemAccessList = []
        currentElemAccessList.append(_elem.name)

        if _elem.attributes.size():
            for key in list(_elem.attributes.keys()):
                self.xmlString += ' ' + key + '="' + _elem.attributes[key] + '"'
                currentElemAccessList.append(key)
                currentElemAccessList.append(_elem.attributes[key])

        self.accesspath.append(currentElemAccessList)

        self.lineToElem[self.lineNumber] = _elem
        self.lineToAccessPath[self.lineNumber] = copy.deepcopy(self.accesspath)
        # print dir(_elem.children)
        # print 'len(childElemList)=',len(childElemList)

        if _elem.children.size():
            self.xmlString += '>'
            if _elem.cdata:
                self.xmlString += spaces + _elem.cdata
            self.newline()

            childElemList = XMLUtils.CC3DXMLListPy(_elem.children)
            for childElem in childElemList:
                self.writeXMLElement(childElem, _indent + self.indentOffset)
            self.xmlString += spaces + '</' + _elem.name + '>'
            self.newline()
        else:
            if _elem.cdata:
                self.xmlString += ">" + _elem.cdata
                self.xmlString += '</' + _elem.name + '>'
                self.newline()
                print('_elem.cdata=', _elem.cdata)
            else:
                self.xmlString += '/>'
                self.newline()
        if len(self.accesspath):
            del self.accesspath[-1]
        # print '_elem.cdata=',_elem.cdata
        # return elemStr


class ParameterScanData:
    def __init__(self):
        self.name = ''
        self.previous_name = None
        self.valueType = FLOAT
        self.type = XML_ATTR
        self.accessPath = ''
        # self.minValue=0
        # self.maxValue=0
        # self.steps=1
        self.customValues = []
        self.currentIteration = 0
        self.hash = ''

    def currentValue(self):
        return self.customValues[self.currentIteration]

    def accessPathToList(self):
        path = self.accessPath
        path = path.replace('[[', '')
        path = path.replace(']]', '')
        path = path.replace(' ', '')
        path = path.replace("'", '')
        pathSplit = path.split("],[")

        accessPathList = []
        for pathElem in pathSplit:
            # print 'pathElem=',pathElem
            pathElemList = pathElem.split(",")

            # pathElemList=[elem for elem in pathElem]
            # print 'pathElemList=',pathElemList
            accessPathList.append(pathElemList)
        return accessPathList
        # print 'accessPath=',accessPath

    def calculateValues(self):
        if self.steps > 1:
            interval = (self.maxValue - self.minValue) / float(self.steps - 1)
            self.customValues = [self.minValue + i * interval for i in range(self.steps)]
        else:
            self.customValues = [(self.maxValue + self.minValue) / 2.0]
        return self.customValues

    def stringHash(self):
        self.accessPath = removeWhiteSpaces(self.accessPath)
        # # print 'str(self.accessPath)=',str(self.accessPath)
        hash = str(self.accessPath) + '_Type=' + TYPE_DICT[self.type] + '_Name=' + self.name
        # # print 'hash=',hash    
        return hash

    def fromXMLElem(self, _el):
        from . import XMLUtils
        self.name = _el.getAttribute('Name')
        # # print 'self.name=',self.name
        self.type = _el.getAttribute('Type')
        self.type = TYPE_DICT_REVERSE[self.type]  # changing string to number to be consistent
        self.valueType = _el.getAttribute('ValueType')
        self.valueType = VALUE_TYPE_DICT_REVERSE[self.valueType]  # changing string to number to be consistent
        self.currentIteration = int(str(_el.getAttribute('CurrentIteration')))
        self.accessPath = removeWhiteSpaces(_el.getText())
        valueStr = str(_el.getFirstElement('Values').getText())

        values = []
        if len(valueStr):
            if self.valueType == STRING:
                values = extractListOfStrings(valueStr)
            else:
                values = valueStr.split(',')

        if len(values):
            if self.valueType == FLOAT:
                self.customValues = list(map(float, values))
            elif self.valueType == INT:
                self.customValues = list(map(int, values))
            else:
                self.customValues = values

        else:
            self.customValues = []

        # # print 'self.customValues=',self.customValues

    def toXMLElem(self):
        from . import XMLUtils
        from .XMLUtils import ElementCC3D

        el = ElementCC3D('Parameter',
                         {'Name': self.name, 'Type': TYPE_DICT[self.type], 'ValueType': VALUE_TYPE_DICT[self.valueType],
                          'CurrentIteration': self.currentIteration}, removeWhiteSpaces(self.accessPath))

        valStr = ''

        if self.valueType == STRING:
            for val in self.customValues:
                valStr += '"' + str(val) + '",'
        else:
            for val in self.customValues:
                valStr += str(val) + ','

        # remove last comma
        if valStr:
            valStr = valStr[:-1]

        el.ElementCC3D('Values', {}, valStr)

        return el


class ParameterScanUtils:

    def __init__(self):

        self.parameter_scan_specs = {
            'version': '4.0.0',
            'parameter_list': {

            }

        }
        self.parameter_scan_specs_fname = None

    def get_parameter_scan_data_dict(self, key: str) ->dict:
        """
        returns a dictionary corresponding to entry ['parameter_list'][key] in  self.parameter_scan_specs
        :param key:
        :return:
        """
        with contextlib.suppress(KeyError):
            psd = self.parameter_scan_specs['parameter_list'][key]
            return psd

    def remove_from_param_scan(self, key):
        """
        Removes parameter scan entry from self.parameter_scan_specs
        :param key:
        :return:
        """
        with contextlib.suppress(KeyError):
            del self.parameter_scan_specs['parameter_list'][key]

        self.write_parameter_scan_specs(fname=self.parameter_scan_specs_fname)

    def addParameterScanData(self, psd: ParameterScanData, original_value: Optional[str] = None):
        """
        Adds parameter scan specification (for a single scanned parameter) to the dictionary of all  scanned parameters
        :param psd:
        :param original_value
        :return:
        """
        try:
            parameter_list_dict = self.parameter_scan_specs['parameter_list']
        except KeyError:
            raise ValueError('<b>parameter_list</b> key is missing from parameter_scan_specs dictionary. '
                             'This indicates corrupt <b>ParameterScan.json</b> file. Please delete content of this '
                             'file and start again')

        modifying_existing_parameter = False
        if psd.previous_name is not None and psd.name != psd.previous_name:
            modifying_existing_parameter = True

        if not modifying_existing_parameter and psd.name in parameter_list_dict.keys():
            raise ValueError(f'You are requesting to add parameter <b>{psd.name}</b> but this identifier already '
                             f'exists in the parameter scan specs')

        if modifying_existing_parameter:
            try:
                del parameter_list_dict[psd.previous_name]
            except KeyError:
                raise ValueError(f'Could not find entry for the previous name <b>{psd.previous_name}</b> '
                                 f'of the parameter {psd.name}')

        parameter_list_dict[psd.name] = {
            'original_value': original_value if original_value is not None else '',
            'values': psd.customValues
        }

        self.write_parameter_scan_specs(fname=self.parameter_scan_specs_fname)

    def refreshParamSpecsContent(self, _pScanFileName):
        self.readParameterScanSpecs(_pScanFileName)

    def readParameterScanSpecs(self, fname):

        pth = Path(fname)
        with pth.open('r') as json_in:
            self.parameter_scan_specs = json.load(json_in)
            self.parameter_scan_specs_fname = pth

    def write_parameter_scan_specs(self, fname):

        with Path(fname).open('w') as json_out:
            json.dump(self.parameter_scan_specs, json_out, indent=4)
            self.parameter_scan_specs_fname = fname

