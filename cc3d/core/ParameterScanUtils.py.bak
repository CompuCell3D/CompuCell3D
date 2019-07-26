from __future__ import with_statement
# enabling with statement in python 2.5

    
import os
import sys
import XMLUtils
# from CC3DProject.enums import *
from ParameterScanEnums import *
from OrderedDict import OrderedDict


def removeWhiteSpaces(_str):
    import re
    out_str=str(_str)
    pattern = re.compile(r'\s+')
    out_str = re.sub(pattern, '', out_str)
    return out_str 
    
def extractListOfStrings(_valueStr):
    values=None
    pos=_valueStr.find('"')     # searching for " if found this means we have composite strings with commas inside - hence we cannot split the string based on comma position       
    if pos>=0:
        values=_valueStr.split('",')
        for i in range(len(values)):
            val=values[i]
            val=val.replace('"','')
            values[i]=val            
    else:
        values=_valueStr.split('",')
    return values    
    

def nextIterationCartProd(currentIteration,maxVals):
    '''Computes next iteration of cartesian product
    '''
    length=len(currentIteration)
    import copy
    
    nextIteration=copy.deepcopy(currentIteration)
    
    
    
    # print 'currentIteration=',currentIteration
    #determine the lowest index with currentIteration value less than max value 
    idx=-1
    for i in range(length):
        if currentIteration[i]<maxVals[i]:
            idx=i
            break
            
    if idx==-1:
        return [0 for i in range(length) ] # rollover
        # raise StopIteration
        
    #increment currentIteration for idx and zero iterations for i<idx 
    nextIteration[idx]+=1
    # print 'currentIteration=',currentIteration
    
    for i in range(idx):
        nextIteration[i]=0
    
    return nextIteration    

def getParameterScanCommandLineArgList(_simFileName):
    '''returns command line options for parameter scan WITHOUT actual run script. run script has to be fetched independently using getCC3DPlayerRunscriptPath or getCC3DPlayerRunscriptPath in SystemUtils
    '''
    import sys
    # # # from SystemUtils import getCC3DPlayerRunscriptPath
                    
    # # # cc3dPath=getCC3DPlayerRunscriptPath()
    reminderArgs=sys.argv[1:-1] # w skip first and last arguments 
    # print 'reminderArgs=',reminderArgs

    # check if arg -i <simulation name> exist
    try:
        idx=reminderArgs.index('-i')                    
        # # # reminderArgs=reminderArgs[0:idx]+reminderArgs[idx+2:]
    except ValueError,e:
        # if -i <simulationName> does not exist we add it to command line
        reminderArgs=['-i',_simFileName]+reminderArgs
        
    # cmdLineArgs=[cc3dPath]+reminderArgs
    cmdLineArgs=reminderArgs
    return cmdLineArgs
    
    

class XMLHandler:
    def __init__(self):
        self.lineNumber=0
        self.accesspath=[]
        self.currentElement=None
        self.indent=0
        self.indentOffset=4
        self.xmlString=''
        self.lineToElem={}
        self.lineToAccessPath={}
        self.accessPathList=[]
        self.cc3dXML2ObjConverter=None
        self.root_element=None
        
    def newline(self):
        self.xmlString+='\n'
        self.lineNumber+=1
        
    def outputXMLWithAccessPaths(self,_xmlFilePath): 
        import XMLUtils
        import os
        self.cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
                
        self.root_element = self.cc3dXML2ObjConverter.Parse(_xmlFilePath)
        xmlStr=self.writeXMLElement(self.root_element)
        # print 'xmlStr=',self.xmlString
    
    def writeXMLElement(self,_elem,_indent=0):
        self.indent=_indent
        import XMLUtils          
        import copy    
        spaces=' '*self.indent
        
        self.xmlString+=spaces+'<'+_elem.name
        currentElemAccessList=[]
        currentElemAccessList.append(_elem.name)                   
        
        if _elem.attributes.size():
            for key in _elem.attributes.keys():
                self.xmlString+=' '+key+'="'+_elem.attributes[key]+'"'            
                currentElemAccessList.append(key)
                currentElemAccessList.append(_elem.attributes[key])
                
        self.accesspath.append(currentElemAccessList)
        
        self.lineToElem[self.lineNumber]=_elem
        self.lineToAccessPath[self.lineNumber]=copy.deepcopy(self.accesspath)
        # print dir(_elem.children)
        # print 'len(childElemList)=',len(childElemList)
        
        if _elem.children.size():
            self.xmlString+='>'
            if _elem.cdata:
                self.xmlString+=spaces+_elem.cdata
            self.newline()                
            
            
            
            childElemList=XMLUtils.CC3DXMLListPy(_elem.children) 
            for childElem in childElemList:
                self.writeXMLElement(childElem,_indent+self.indentOffset)
            self.xmlString+=spaces+'</'+_elem.name+'>'
            self.newline()                
        else:
            if _elem.cdata:                
                self.xmlString+=">"+_elem.cdata
                self.xmlString+='</'+_elem.name+'>'
                self.newline()                
                print '_elem.cdata=',_elem.cdata
            else:        
                self.xmlString+='/>'
                self.newline()
        if len(self.accesspath):
            del self.accesspath[-1]
        # print '_elem.cdata=',_elem.cdata
        # return elemStr
        
 
            
        
class ParameterScanData:
    def __init__(self):        
        self.name=''
        self.valueType=FLOAT
        self.type=XML_ATTR
        self.accessPath=''
        # self.minValue=0
        # self.maxValue=0
        # self.steps=1
        self.customValues=[]
        self.currentIteration=0
        self.hash=''
        
    def currentValue(self):
        return self.customValues[self.currentIteration]
        
    def accessPathToList(self):
        path=self.accessPath
        path=path.replace('[[','')
        path=path.replace(']]','')
        path=path.replace(' ','')
        path=path.replace("'",'')
        pathSplit=path.split("],[")
        

        accessPathList=[]
        for pathElem in pathSplit:
            # print 'pathElem=',pathElem
            pathElemList=pathElem.split(",")

            # pathElemList=[elem for elem in pathElem]
            # print 'pathElemList=',pathElemList
            accessPathList.append(pathElemList)
        return accessPathList
        # print 'accessPath=',accessPath
        
    def calculateValues(self):
        if self.steps>1:
            interval=(self.maxValue-self.minValue)/float(self.steps-1)
            self.customValues=[self.minValue+i*interval for i in range(self.steps)]            
        else:
            self.customValues=[(self.maxValue+self.minValue)/2.0]
        return self.customValues
        
    def  stringHash(self):
        self.accessPath=removeWhiteSpaces(self.accessPath)
        # # print 'str(self.accessPath)=',str(self.accessPath)
        hash=str(self.accessPath)+'_Type='+TYPE_DICT[self.type]+'_Name='+self.name   
        # # print 'hash=',hash    
        return hash
        
    def fromXMLElem(self,_el):
        import XMLUtils
        self.name = _el.getAttribute('Name')
        # # print 'self.name=',self.name
        self.type = _el.getAttribute('Type')
        self.type = TYPE_DICT_REVERSE[self.type] # changing string to number to be consistent
        self.valueType = _el.getAttribute('ValueType')
        self.valueType = VALUE_TYPE_DICT_REVERSE[self.valueType]# changing string to number to be consistent
        self.currentIteration = int(str(_el.getAttribute('CurrentIteration')))
        self.accessPath = removeWhiteSpaces(_el.getText())
        valueStr=str(_el.getFirstElement('Values').getText())
        
        values=[]
        if len(valueStr):            
            if self.valueType==STRING:
                values=extractListOfStrings(valueStr)
            else:
                values=valueStr.split(',')
        
        if len(values):
            if self.valueType==FLOAT:
                self.customValues=map(float,values)
            elif self.valueType==INT:   
                self.customValues=map(int,values)
            else:
                self.customValues=values
                
        else:
            self.customValues=[]
        
        # # print 'self.customValues=',self.customValues
        
    def toXMLElem(self):
        import XMLUtils
        from XMLUtils import ElementCC3D
        
        el=ElementCC3D('Parameter',{'Name':self.name,'Type':TYPE_DICT[self.type],'ValueType':VALUE_TYPE_DICT[self.valueType],'CurrentIteration':self.currentIteration},removeWhiteSpaces(self.accessPath))
        
        valStr=''
        
        if self.valueType==STRING:
            for val in self.customValues:
                valStr+='"'+str(val)+'",'
        else:
            for val in self.customValues:
                valStr+=str(val)+','
            
        #remove last comma    
        if valStr:
            valStr=valStr[:-1]
            
            
        el.ElementCC3D('Values',{},valStr)
        
            
        return el

class ParameterScanUtils:

    def __init__(self):
    
        self.cc3dXML2ObjConverter=None
        self.root_element=None
        self.parameterScanFileToDataMap = OrderedDict() # {file name:dictionary of parameterScanData} parameterScanDataMap={hash:parameterScanData}
        self.outputDirectoryRelativePath=''        
        
    def initialize(self):
        self.cc3dXML2ObjConverter=None
        self.root_element=None
        self.parameterScanFileToDataMap = OrderedDict() # {file name:dictionary of parameterScanData} parameterScanDataMap={hash:parameterScanData}
        self.outputDirectoryRelativePath=''        
        
    def setOutputDirectoryRelativePath(self,_path):
        self.outputDirectoryRelativePath=_path
        
    def extractXMLScannableParameters(self,_elem, _scanFile):
        '''
            returns dictionary of scannable parameters for a given XML file
        '''
        # parse current scan file
        self.xmlHandler=XMLHandler
        
        params={}
        # # print '_elem=',_elem.name
        # # print '_elem.attributes.size()=',_elem.attributes
        
        if _elem.attributes.size():
            for key in _elem.attributes.keys():
                try:# checki if attribute can be converted to floating point value - if so it can be added to scannable parameters
                    print '_elem.attributes[key]=',_elem.attributes[key]
                    float(_elem.attributes[key])
                    params[key]=[_elem.attributes[key],XML_ATTR,FLOAT]
                except ValueError,e:
                    pass
                    
        #check if cdata is a number - if so this could be scannable parameter
        try:# checki if attribute can be converted to floating point value - if so it can be added to scannable parameters

            float(_elem.cdata)
            params[_elem.name]=[_elem.cdata,XML_CDATA,FLOAT]
        except ValueError,e:
            pass
        
        return params        
        
        
    def addParameterScanData(self,_relativePath,_psd):

        
        try:
            parameterScanDataMap=self.parameterScanFileToDataMap[_relativePath]            
        except LookupError,e:
            parameterScanDataMap=OrderedDict()
            self.parameterScanFileToDataMap[_relativePath]=parameterScanDataMap
            
            
        parameterScanDataMap[_psd.stringHash()]=_psd
        
    def parseXML(self,_xmlFile):
        import xml
        
        
        import XMLUtils
        import os
        cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
        try:
            root_element=cc3dXML2ObjConverter.Parse(_xmlFile)    
                        
            return root_element,cc3dXML2ObjConverter
            
        except xml.parsers.expat.ExpatError,e:
            print 'Error Parsing Parameter scan file\n\n\n'
            return None,None        
        
    def refreshParamSpecsContent(self,_pScanFileName):
        self.readParameterScanSpecs(_pScanFileName)
        
    def readParameterScanSpecs(self,_pScanFileName=''):
        
        self.initialize()
        
        if not _pScanFileName:return
        
        root_element,xml2ObjConverter=self.parseXML(_pScanFileName)
                
        
        if not root_element:return
        
        
        self.parameterScanFileToDataMap=OrderedDict()
            
        
        # print 'elem=',elem 
        self.outputDirectoryRelativePath=str(root_element.getFirstElement("OutputDirectory").getText())
                                                                                                                 
        
        
        paramListElemList = XMLUtils.CC3DXMLListPy(root_element.getElements("ParameterList"))
        for paramListElem in paramListElemList:
            filePath=paramListElem.getAttribute('Resource')
            # # print 'filePath=',filePath 
            
            parameterScanDataMap=OrderedDict()
            self.parameterScanFileToDataMap[filePath]=parameterScanDataMap
            
            # # print 'READING self.parameterScanFileToDataMap=',self.parameterScanFileToDataMap
                        
            paramElemList=XMLUtils.CC3DXMLListPy(paramListElem.getElements("Parameter")) 
            
            
            for paramElem in paramElemList:
                # from ParameterScanUtils import ParameterScanData            
                psd=ParameterScanData()                
                psd.fromXMLElem(paramElem)            
                
                #storing psd in the dictionary
                parameterScanDataMap[psd.stringHash()]=psd        
                
        # # print 'AFTER READING PARAM SCAN SPECS ',self.parameterScanFileToDataMap
        # sys.exit()
        
    # def normalizeParameterScanSpecs(self,_pScanFileName):
        # '''This function writes parameter scan file in order set by internal dictionaries of the ParameterScanUtils class. We call it each time we run new parameters scan to ensure that the ordering of dictionaries and file content is the same  
        # '''
         # pass   
        # # self.readParameterScanSpecs(_pScanFileName)
        # # self.writeParameterScanSpecs(_pScanFileName)
        
    def getParameterScanSpecsXMLString(self):
        import XMLUtils
        from XMLUtils import ElementCC3D
        import os    

        root_elem=ElementCC3D('ParameterScan',{'version':'3.7.0'})
        root_elem.ElementCC3D('OutputDirectory',{},self.outputDirectoryRelativePath)
        
        # print 'csd.parameterScanResource.parameterScanXMLElements=',self.parameterScanXMLElements
        
        xmlElemTmpStorage=[]
        
        # # print 'JUST BEFORE WRITING self.parameterScanFileToDataMap=',self.parameterScanFileToDataMap
        
        for fileName, parameterScanDataMap in self.parameterScanFileToDataMap.iteritems():
            if len(parameterScanDataMap.keys()):
                # # print ' adding paramList element =',fileName
                paramListElem=root_elem.ElementCC3D('ParameterList',{'Resource':fileName})
                
                xmlElemTmpStorage.append(paramListElem)
                
                for hash, psd in parameterScanDataMap.iteritems():
                    xmlElem=psd.toXMLElem()
                    xmlElemTmpStorage.append(xmlElem)
                    
                    paramListElem.CC3DXMLElement.addChild(xmlElem.CC3DXMLElement)
                    
        
        return root_elem.CC3DXMLElement.getCC3DXMLElementString()
        
    def writeParameterScanSpecs(self,_pScanFileName):       

        import XMLUtils
        from XMLUtils import ElementCC3D
        import os    

        root_elem=ElementCC3D('ParameterScan',{'version':'3.7.0'})
        root_elem.ElementCC3D('OutputDirectory',{},self.outputDirectoryRelativePath)
        
        # print 'csd.parameterScanResource.parameterScanXMLElements=',self.parameterScanXMLElements
        
        xmlElemTmpStorage=[]
        
        # # print 'JUST BEFORE WRITING self.parameterScanFileToDataMap=',self.parameterScanFileToDataMap
        
        for fileName, parameterScanDataMap in self.parameterScanFileToDataMap.iteritems():
            if len(parameterScanDataMap.keys()):
                # # print ' adding paramList element =',fileName
                paramListElem=root_elem.ElementCC3D('ParameterList',{'Resource':fileName})
                
                xmlElemTmpStorage.append(paramListElem)
                
                for hash, psd in parameterScanDataMap.iteritems():
                    xmlElem=psd.toXMLElem()
                    xmlElemTmpStorage.append(xmlElem)
                    
                    paramListElem.CC3DXMLElement.addChild(xmlElem.CC3DXMLElement)    
                    
        root_elem.CC3DXMLElement.saveXML(_pScanFileName) 
        
        
    def resetParameterScan(self,_pScanFileName):
        '''This function resets state of the parameter scan to the beginning
        '''


        self.readParameterScanSpecs(_pScanFileName)
        for filePath,parameterScanDataMap in self.parameterScanFileToDataMap.items():
            for hash,psd in parameterScanDataMap.items():
                psd.currentIteration=0          
        
        self.writeParameterScanSpecs(_pScanFileName)    
    
    def prepareParameterScanOutputDirs(self,_outputDirRoot):
        import os
        pScanOutputDirRelPath=self.outputDirectoryRelativePath
        
        if pScanOutputDirRelPath=='':
            pScanOutputDirRelPath='ParameterScan'
            
        customPath=os.path.join(_outputDirRoot,pScanOutputDirRelPath)    
        
        iterationId=self.computeCurrentIterationIdNumber()
        # print 'param SCAN XML=',self.getParameterScanSpecsXMLString()
        
        print '\n\n\n iterationId=',iterationId
        
        customOutputPath=os.path.join(_outputDirRoot,pScanOutputDirRelPath)
        customOutputPath=os.path.join(customOutputPath,str(iterationId))        
        
        #make output directory    
        
        try:
            os.makedirs(customOutputPath)
        except :
            print 'COULD NOT WRITE customOutputPath=',customOutputPath
            # return None
            raise AssertionError('Parameter Scan ERRORCODE='+str(SCAN_FINISHED_OR_DIRECTORY_ISSUE)+': Could not create directory '+customOutputPath+ ' . It is likely that parameter scan has finished running or you may have permission issues preventing directory creation. If it is the latter, please make sure you have necessary write permissions or choose another directory')
            # print 'Could not create directory ',customOutputPath, ' . please make sure you have necessary write permissions'
            
            
        return customOutputPath
        
    
    def reachedMaxIteration(self):
        maxVals=[] # max values for parameter scans
        currentIteration=[] # current iteration
        
        for filePath,parameterScanDataMap in self.parameterScanFileToDataMap.items():
            for hash,psd in parameterScanDataMap.items():
                maxVals.append(len(psd.customValues)-1)
                currentIteration.append(psd.currentIteration)
                
        return True if currentIteration==maxVals else False
            
        
    def computeNextIteration(self):
        '''given current state of the parameter scan computes next combination of the parameters for the parameter scan 
        '''
        
        maxVals=[] # max values for parameter scans
        currentIteration=[] # current iteration
        
        for filePath,parameterScanDataMap in self.parameterScanFileToDataMap.items():
            for hash,psd in parameterScanDataMap.items():
                maxVals.append(len(psd.customValues)-1)
                currentIteration.append(psd.currentIteration)
                
        print 'maxVals=',maxVals
        print 'currentIteration=',currentIteration
        
        iteration=nextIterationCartProd(currentIteration,maxVals)
        
        return iteration
        
    def writeParameterScanSpecsWithIteration(self,_pScanFileName,_iteration):    
        '''Saves parameter scan spec file with updated iteration state. dictionary traversal must be done in an identical way as in computeNextIteration function . these two functions have to be in sync
        '''
        counter=0
        print '\n\n\n writeParameterScanSpecsWithIteration self.parameterScanFileToDataMap=',self.parameterScanFileToDataMap
        for filePath,parameterScanDataMap in self.parameterScanFileToDataMap.items():
            for hash,psd in parameterScanDataMap.items():                
                psd.currentIteration=_iteration[counter]
                counter+=1
                
        self.writeParameterScanSpecs(_pScanFileName)
        
    def saveParameterScanState(self,_pScanFileName ):        
        
        self.refreshParamSpecsContent(_pScanFileName)
        iteration=self.computeNextIteration()        
        self.writeParameterScanSpecsWithIteration(_pScanFileName,iteration)        
        
            
        
        
    def  computeCurrentIterationIdNumber(self):            
        
        
        currentIteration=[] # current iteration
        
        for filePath,parameterScanDataMap in self.parameterScanFileToDataMap.items():
            for hash,psd in parameterScanDataMap.items():                
                currentIteration.append(psd.currentIteration) 
                
        return  self.computeIterationIdNumber(currentIteration)      
    
    def computeIterationIdNumber(self,_iteration):
        maxValsMultiply=[] # max values for parameter scans - increased by one for multiplication purposes
        currentIteration=_iteration
        
        
        
        for filePath,parameterScanDataMap in self.parameterScanFileToDataMap.items():
            for hash,psd in parameterScanDataMap.items():
                maxValsMultiply.append(len(psd.customValues))                

        multiplicativeFactors=[1 for i in currentIteration]   
        
        
        for i in range(1,len(multiplicativeFactors)):
            multiplicativeFactors[i]=multiplicativeFactors[i-1]*maxValsMultiply[i-1]
        
        print 'currentIteration=',currentIteration
        print 'multiplicativeFactors=',multiplicativeFactors
        id=0
        for i in range(len(multiplicativeFactors)):
            id+=multiplicativeFactors[i]*currentIteration[i]
        
        return id
        

        
    def  getXMLElementFromAccessPath(self,_root_element, _accessPathList):
        ''' This fcn fetches xml element 
            This Function greatly simplifies access to XML data - one line  easily replaces  many lines of code
        '''
        # import types                
        
        from  itertools import izip
        from XMLUtils import dictionaryToMapStrStr as d2mss
        
        tmpElement=_root_element
        attrDict=None
        for arg in _accessPathList[1:]: # we skip first element of the acces path as it points to root_element
            #constructing attribute dictionary
            
            if len(arg)>=3:
                attrDict={}
                for tuple in izip(arg[1::2],arg[2::2]):                            
                    if attrDict.has_key(tuple[0]):
                        raise LookupError ('Duplicate attribute name in the access path '+str(_accessPathList))
                    else:
                        attrDict[tuple[0]]=tuple[1]                        
                attrDict=d2mss(attrDict)
                # attrDict=d2mss(dict((tuple[0],tuple[1]) for tuple in izip(arg[1::2],arg[2::2])))
                

            elemName=arg[0]        
            # print 'elemName=',elemName
            # print 'attrDict=',attrDict
            tmpElement=tmpElement.getFirstElement(arg[0],attrDict) if attrDict is not None else tmpElement.getFirstElement(arg[0])
            attrDict = None
            # tmpElement=tmpElement.getFirstElement(arg[0],attrDict)
        
        return tmpElement        
        
    def  getXMLElementValue(self, _root_element, _accessPathList):
        
        element=self.getXMLElementFromAccessPath(_root_element, _accessPathList)        
        return element.getText() if element else None        
    
    def  setXMLElementValue(self, _root_element, _accessPathList,_value):    

        element=self.getXMLElementFromAccessPath(_root_element,_accessPathList)    
        if element:    
            element.updateElementValue(str(_value))            

            
    def  getXMLAttributeValue(self, _root_element, _accessPathList, _attr):
    
        element=self.getXMLElementFromAccessPath(_root_element, _accessPathList)    
        
        if  element is not None:
            if element.findAttribute(_attr):
                return element.getAttribute(_attr)
            else:
                raise LookupError ('Could not find attribute '+_attr+' in '+_accessPathList)
        else:
            return None
            
    def  setXMLAttributeValue(self,_root_element,_accessPathList,_attr,_value):    
    
        element=self.getXMLElementFromAccessPath(_root_element,_accessPathList) 
        
        if element:    
            if element.findAttribute(_attr):                
                from XMLUtils import dictionaryToMapStrStr as d2mss
                element.updateElementAttributes(d2mss({_attr:_value}))            
        
    def  replaceValuesInXMLFile(self,_parameterScanDataMap,_xmlFile):
        # print 'processing XML'
        root_element, xml2ObjConverter=self.parseXML(_xmlFile)
        for hash,psd in _parameterScanDataMap.items():
            # print 'psd.accessPath=',psd.accessPath
            accessPathList=psd.accessPathToList()
            
            elem=self.getXMLElementFromAccessPath(root_element,accessPathList) 
            if elem:
                if psd.type==XML_CDATA:
                    self.setXMLElementValue(root_element,accessPathList,psd.currentValue())
                elif psd.type==XML_ATTR:
                    self.setXMLAttributeValue(root_element,accessPathList, psd.name, psd.currentValue())
                    
            # print 'accessPathList=',accessPathList
            # print 'elem=',elem
            # print 'customValue=',psd.currentValue()
            
        root_element.saveXML(_xmlFile)
        
    def checkPythonLineForGlobalVariable(self,_line , _variableName=''):
        import re
        globalPythonVarRegex=None
        if  not _variableName:            
            globalPythonVarRegex=re.compile("[\S]*[\s]*=") 
        else: 
            globalPythonVarRegex=re.compile(_variableName+"[\s]*=")        
        matchObj=re.match(globalPythonVarRegex,_line)                    
        
        return True if matchObj else False 


    def extractGlobalVarFromLine(self,_line , _variableName=''):
        import re
        globalPythonVarRegex=None
        if  not _variableName:            
            globalPythonVarRegex=re.compile("([\S]*)[\s]*=[\s]*([\S]*)") 
        else: 
            globalPythonVarRegex=re.compile("("+_variableName+")"+"[\s]*=[\s]*([\S]*)")        
        matchObj=re.match(globalPythonVarRegex,_line)         
        
        return matchObj.group(1),matchObj.group(2)
        # return True if matchObj else False 

        
    def  replaceValuesInPythonFile(self,_parameterScanDataMap,_pythonFile):    
        fileContent=[]
        
        
        for line in open(_pythonFile):
            fileContent.append(line)
                
        import re    
               
        for hash,psd in _parameterScanDataMap.items():
            
            if psd.type==PYTHON_GLOBAL:
            
                for idx in range(len(fileContent)):
                
                    line = fileContent[idx]
                    
                    foundGlobalVar=self.checkPythonLineForGlobalVariable(line,psd.name)
                    
                    if  foundGlobalVar:
                    
                        if psd.valueType==STRING:
                            # print 'GOT STRING VALUE FOR ',psd.name
                            # if psd.name=='MYVAR1':
                                # print 'psd.currentValue()=',psd.currentValue()                                
                                
                            fileContent[idx]=psd.name+" = '"+str(psd.currentValue())+"'\n"                           
                        else:
                            fileContent[idx]=psd.name+' = '+str(psd.currentValue())+'\n'
                        break
                        
            else:
                raise AssertionError('ParameterScans: Trying to replace parameter which is not of PYTHON_GLOBAL type')
                
        with open(_pythonFile,'w') as f:
            for line in fileContent: print >>f,line, # last comma is to avoid autiomatic insertion of the new line
                
        
    def replaceValuesInSimulationFiles(self,_pScanFileName='',_simulationDir=''):
        
        self.readParameterScanSpecs(_pScanFileName)
        for filePath,parameterScanDataMap in self.parameterScanFileToDataMap.items():
            fullFilePath=os.path.join(_simulationDir,filePath)
            
            if os.path.splitext(filePath)[1].lower()=='.xml':
                self.replaceValuesInXMLFile(_parameterScanDataMap = parameterScanDataMap , _xmlFile=fullFilePath)               
            elif os.path.splitext(filePath)[1].lower()=='.py':                
                self.replaceValuesInPythonFile(_parameterScanDataMap = parameterScanDataMap , _pythonFile=fullFilePath)
                
            else:
                raise  AssertionError('Can only replace values in the XML and Python files. please make sure file extensions are .xml or .py')
                
                
                
        
        
        
    
        
                
            