import os
import sys
import XMLUtils
from CC3DProject.enums import *

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
        print 'xmlStr=',self.xmlString
    
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
        
def removeWhiteSpaces(_str):
    import re
    out_str=str(_str)
    pattern = re.compile(r'\s+')
    out_str = re.sub(pattern, '', out_str)
    return out_str 
    
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
    
    def calculateValues(self):
        if self.steps>1:
            interval=(self.maxValue-self.minValue)/float(self.steps-1)
            self.customValues=[self.minValue+i*interval for i in range(self.steps)]
            
        else:
            self.customValues=[(self.maxValue+self.minValue)/2.0]
        return self.customValues
        
    def  stringHash(self):
        self.accessPath=removeWhiteSpaces(self.accessPath)
        print 'str(self.accessPath)=',str(self.accessPath)
        hash=str(self.accessPath)+'_Type='+TYPE_DICT[self.type]+'_Name='+self.name   
        print 'hash=',hash    
        return hash
        
    def fromXMLElem(self,_el):
        import XMLUtils
        self.name = _el.getAttribute('Name')
        print 'self.name=',self.name
        self.type = _el.getAttribute('Type')
        self.type = TYPE_DICT_REVERSE[self.type] # changing string to number to be consistent
        self.valueType = _el.getAttribute('ValueType')
        self.valueType = VALUE_TYPE_DICT_REVERSE[self.valueType]# changing string to number to be consistent
        self.currentIteration = str(_el.getAttribute('CurrentIteration'))
        self.accessPath = removeWhiteSpaces(_el.getText())
        valueStr=str(_el.getFirstElement('Values').getText())
        
        values=[]

        if len(valueStr):            
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
        
    def toXMLElem(self):
        import XMLUtils
        from XMLUtils import ElementCC3D
        
        el=ElementCC3D('Parameter',{'Name':self.name,'Type':TYPE_DICT[self.type],'ValueType':VALUE_TYPE_DICT[self.valueType],'CurrentIteration':self.currentIteration},removeWhiteSpaces(self.accessPath))
        
        valStr=''
        # values=self.calculateValues()
        
        for val in self.customValues:
            valStr+=str(val)+','
            
        #remove last comma    
        if valStr:
            valStr=valStr[:-1]
            
            
        el.ElementCC3D('Values',{},valStr)
        
        # if len(self.customValues):
            
            # for val in self.customValues:
                # el.ElementCC3D('CustomValue',{},val)
        # else:        
            # el.ElementCC3D('MinValue',{},self.minValue)
            # el.ElementCC3D('MaxValue',{},self.maxValue)
            # el.ElementCC3D('Steps',{},self.steps)
            
        return el

class ParameterScanUtils:

    def __init__(self):
    
        self.cc3dXML2ObjConverter=None
        self.root_element=None
        
    def parseScanFile(self, _scanFile):
        import XMLUtils
        import os
        self.cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
                
        self.root_element = self.cc3dXML2ObjConverter.Parse(_xmlFilePath)
        
        paramElemList=XMLUtils.CC3DXMLListPy(self.root_element) 
        for elem in paramElemList:
            print 'elem.name=',elem.name

        
        
    def extractXMLScannableParameters(self,_elem, _scanFile):
        '''
            returns dictionary of scannable parameters for a given XML file
        '''
        # parse current scan file
        self.xmlHandler=XMLHandler
        
        params={}
        print '_elem=',_elem.name
        print '_elem.attributes.size()=',_elem.attributes
        
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
        
        