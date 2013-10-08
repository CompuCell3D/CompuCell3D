import os
import sys
import XMLUtils

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

class ParameterScanData:
    def __init__(self):
        self.name=''
        self.type=''
        self.accessPath=''
        self.minValue=0
        self.maxValue=0
        self.steps=1
        self.customValues=[]
        
    def toXMLElem(self):
        import XMLUtils
        from XMLUtils import ElementCC3D
        
        el=ElementCC3D('Parameter',{'Name':self.name,'Type':self.type},self.accessPath)
        
        if len(self.customValues):
            for val in self.customValues:
                el.ElementCC3D('CustomValue',{},val)
        else:        
            el.ElementCC3D('MinValue',{},self.minValue)
            el.ElementCC3D('MaxValue',{},self.maxValue)
            el.ElementCC3D('Steps',{},self.steps)
            
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
                    params[key]=[_elem.attributes[key],0]
                except ValueError,e:
                    pass
                    
        #check if cdata is a number - if so this could be scannable parameter
        try:# checki if attribute can be converted to floating point value - if so it can be added to scannable parameters

            float(_elem.cdata)
            params[_elem.name]=[_elem.cdata,0]
        except ValueError,e:
            pass
        
        return params        
        
        