# -*- coding: utf-8 -*-
import os, sys
import string

MODULENAME = '------- pythonSetupScripts/CC3DSimulationDataHandler.py: '

def findRelativePathSegments(basePath,p, rest=[]):
    """
        This function finds relative path segments of path p with respect to base path    
        It returns list of relative path segments and flag whether operation succeeded or not    
    """
    h,t = os.path.split(p)
    pathMatch=False
    if h==basePath:
        pathMatch=True
        return [t]+rest,pathMatch
    print "(h,t,pathMatch)=",(h,t,pathMatch)
    if len(h) < 1: return [t]+rest,pathMatch
    if len(t) < 1: return [h]+rest,pathMatch
    return findRelativePathSegments(basePath,h,[t]+rest)
    
def findRelativePath(basePath,p):
    relativePathSegments,pathMatch = findRelativePathSegments(basePath,p)
    if pathMatch:
        relativePath=""
        for i in range(len(relativePathSegments)):
            segment = relativePathSegments[i]
            relativePath += segment
            if i != len(relativePathSegments)-1:
                relativePath += "/" # we use unix style separators - they work on all (3) platforms
        return relativePath
    else:
        return p


class GenericResource(object):
    def __init__(self,_resourceName=''):
        self.resourceName=_resourceName

class CC3DResource(GenericResource):
    def __init__(self):
        GenericResource.__init__(self,'CC3DResource')
        self.path=""
        self.type=""
        self.module=""
        self.origin="" # e.g. serialized
        self.copy=True
    def __str__(self):
        return "ResourcePath: "+str(self.path)+"\n"+"Type: "+str(self.type)+"\n"+"Module: "+str(self.module)+"\n"+"Origin: "+str(self.origin)+"\n"
    def __repr__(self):
        return self.__str__()
        
class CC3DSerializerResource(GenericResource):
    def __init__(self):        
        GenericResource.__init__(self,'CC3DSerializerResource')
        self.outputFrequency=0
        self.allowMultipleRestartDirectories=False
        self.fileFormat='text'    
        self.restartDirectory=''
        
    def disableRestart(self):
        self.restartDirectory=''
            
    def enableRestart(self,_restartDir=''):
        if _restartDir!='':
            self.restartDirectory=_restartDir
        else:
            self.restartDirectory='restart'
        
    def appendXMLStub(self, _rootElem):     
        from XMLUtils import ElementCC3D  
        print MODULENAME,'IN APPEND XML STUB'
        print 'self.restartDirectory=',self.restartDirectory
        
        if self.outputFrequency>0:
            attributeDict={"OutputFrequency":self.outputFrequency,"AllowMultipleRestartDirectories":self.allowMultipleRestartDirectories,"FileFormat":self.fileFormat}
            _rootElem.ElementCC3D('SerializeSimulation',attributeDict)            
        if self.restartDirectory != '':
            attributeDict={"RestartDirectory":self.restartDirectory}
            _rootElem.ElementCC3D('RestartSimulation',attributeDict)
            print MODULENAME,'attributeDict=',attributeDict

            
from ParameterScanUtils import ParameterScanUtils       
     
class CC3DParameterScanResource(CC3DResource):
    def __init__(self):        
        CC3DResource.__init__(self)
        self.resourceName='CC3DParameterScanResource'
        self.type='ParameterScan'
        self.basePath=''
        
        self.parameterScanXMLElements = {}
        self.parameterScanFileToDataMap = {} # {file name:dictionary of parameterScanData} parameterScanDataMap={hash:parameterScanData}
        # self.parameterScanDataMap = {}
        self.fileTypeForEditor = 'xml'
        self.parameterScanXMLHandler = None
        # self.parameterScanEditor=None
        
        self.psu=ParameterScanUtils() # ParameterScanUtils is the class where all parsing and parameter scan data processing takes place
        
    def addParameterScanData(self,_file,_psd):
        print 'self.basePath=',self.basePath
        print '_file=',_file
        relativePath=findRelativePath(self.basePath,_file) # relative path of the scanned simulation file w.r.t. project directory
        print 'relativePath=',relativePath
        self.psu.addParameterScanData(relativePath,_psd)
        
    def readParameterScanSpecs(self):
        self.psu.readParameterScanSpecs(self.path)

    def writeParameterScanSpecs(self):
        self.psu.writeParameterScanSpecs(self.path)
        
        
        
            
class CC3DSimulationData:
    def __init__(self):
        self.pythonScript=""
        self.xmlScript=""
        self.pifFile=""
        self.windowScript=""
        
        self.pythonScriptResource = CC3DResource()
        self.xmlScriptResource = CC3DResource()
        self.pifFileResource = CC3DResource()
        self.windowScriptResource = CC3DResource()
        self.windowDict = {}
        
        self.serializerResource = None
        self.parameterScanResource = None
        
        self.resources={} # dictionary of resource files with description (types, plugin, etc)
        self.path="" # full path to project file
        self.basePath="" # full path to the directory of project file
        
        self.version="3.5.1"
        
    def __str__(self):
        return "CC3DSIMULATIONDATA: "+self.basePath+"\n"+"\tpython file: "+self.pythonScript+"\n"+"\txml file: "+self.xmlScript+"\n"+\
        "\tpifFile="+self.pifFile+"\n"+"\twindow script="+self.windowScript + str(self.resources)
        
    def addNewParameterScanResource(self):
        self.parameterScanResource=CC3DParameterScanResource()
        self.parameterScanResource.path=os.path.abspath(os.path.join(self.basePath,'Simulation/ParameterScanSpecs.xml'))
        
        baseCoreName , ext=os.path.splitext(os.path.basename(self.path)) # extracting core simulation name from full cc3d project path
        
        self.parameterScanResource.psu.setOutputDirectoryRelativePath(baseCoreName+'_ParameterScan')
        
        
    def removeParameterScanResource(self):        
        self.parameterScanResource=None    
    

    
    def addNewSerializerResource(self,_outFreq=0,_multipleRestartDirs=False,_format='text',_restartDir=''):
        self.serializerResource = CC3DSerializerResource()
        
        self.serializerResource.serializerOutputFrequency = _outFreq
        self.serializerResource.serializerAllowMultipleRestartDirectories = _multipleRestartDirs
        self.serializerResource.serializerFileFormat = _format    
        self.serializerResource.restartDirectory = _restartDir
        
    def getResourcesByType(self,_type):
        resourceList=[]
        for key,resource in self.resources.iteritems():
            if resource.type==_type:
                resourceList.append(resource)
        return resourceList
        
    def restartEnabled(self):
        if self.serializerResource:
            return self.serializerResource.restartDirectory !=''
        else:
            return False

    def removeSerializerResource(self):        
        self.serializerResource=None
    
    def addNewResource(self,_fileName,_type):   # called by Twedit
        print 'type(_fileName)=',type(_fileName)
        if _type=="XMLScript":
            self.xmlScript = os.path.abspath(_fileName)
            self.xmlScriptResource.path = self.xmlScript
            self.xmlScriptResource.type = "XML"
            return 
        
        if _type=="PythonScript":
            self.pythonScript = os.path.abspath(_fileName)
            self.pythonScriptResource.path = self.pythonScript
            self.pythonScriptResource.type = "Python"
            return 
            
        if _type=="PIFFile":        
            # we have to check if there is  PIF file assigned resource. If so we do not want to modify 
            # this resource, rather add another one as a generic type of resource
            if self.pifFileResource.path=='':
                self.pifFile = os.path.abspath(_fileName)
                self.pifFileResource.path = self.pifFile
                self.pifFileResource.type = "PIFFile"
            # we will also add PIF File as generic resource
        
        # adding generic resource type - user specified        
        fullPath = os.path.abspath(_fileName)
        resource = CC3DResource()
        resource.path = fullPath
        resource.type = _type
        self.resources[fullPath] = resource
        print MODULENAME        
        print "self.resources=",self.resources
        return 
        
    def removeResource(self,_fileName):
        fileName=os.path.abspath(_fileName)
        print 'TRYING TO REMOVE RESOURCE _fileName=',_fileName
        # file name can be associated with many resources - we have to erase all such associations
        if fileName==self.xmlScript:
            self.xmlScript=""
            self.xmlScriptResource = CC3DResource()
            
        if fileName==self.pythonScript:
            self.pythonScript=""
            self.pythonScriptResource = CC3DResource()
            
        if fileName==self.pifFile:
            self.pifFile=""
            self.pifFileResource = CC3DResource()

        try:
            del self.resources[fileName]
        except LookupError,e:
            pass
            
        print 'After removing resources'    
        
        print  self.resources   
        return 
            
class CC3DSimulationDataHandler:
    def __init__(self,_tabViewWidget=None):
        self.cc3dSimulationData = CC3DSimulationData()
        self.tabViewWidget = _tabViewWidget
        
    def copySimulationDataFiles(self,_dir):
        import shutil        
        simulationPath = os.path.join(_dir,'Simulation')
        
        if not os.path.exists(simulationPath):
            os.makedirs(simulationPath)
            
        #copy project file        
        try:            
            shutil.copy(self.cc3dSimulationData.path,os.path.join(_dir,os.path.basename(self.cc3dSimulationData.path))) 
        except: # ignore any copy errors
            pass
        
        if self.cc3dSimulationData.pythonScript!="":
            shutil.copy(self.cc3dSimulationData.pythonScript,os.path.join(simulationPath,os.path.basename(self.cc3dSimulationData.pythonScript))) 
            
        if self.cc3dSimulationData.xmlScript!="":
            shutil.copy(self.cc3dSimulationData.xmlScript,os.path.join(simulationPath,os.path.basename(self.cc3dSimulationData.xmlScript))) 
        
        if self.cc3dSimulationData.pifFile!="":
            shutil.copy(self.cc3dSimulationData.pifFile,os.path.join(simulationPath,os.path.basename(self.cc3dSimulationData.pifFile))) 
            
        if self.cc3dSimulationData.windowScript!="":
            shutil.copy(self.cc3dSimulationData.windowScript,os.path.join(simulationPath,os.path.basename(self.cc3dSimulationData.windowScript))) 
            
        if self.cc3dSimulationData.parameterScanResource:
            
            shutil.copy(self.cc3dSimulationData.parameterScanResource.path , os.path.join(simulationPath,os.path.basename(self.cc3dSimulationData.parameterScanResource.path))) 

            
        #copy resource files
        fileNames = self.cc3dSimulationData.resources.keys()
        
        for fileName in fileNames:            
            try:
                if self.cc3dSimulationData.resources[fileName].copy:    
                    shutil.copy(fileName,os.path.join(simulationPath,os.path.basename(fileName))) 
            except:
                #ignore any copy errors
                pass
    
    def readCC3DFileFormat(self,_fileName):        
#        print MODULENAME,'  readCC3DFileFormat()'
        import XMLUtils
        xml2ObjConverter = XMLUtils.Xml2Obj()

        fileFullPath = os.path.abspath(_fileName)
        self.cc3dSimulationData.basePath = os.path.dirname(fileFullPath)
        self.cc3dSimulationData.path = fileFullPath
        bp = self.cc3dSimulationData.basePath
        root_element = xml2ObjConverter.Parse(fileFullPath) # this is simulation element
        
        if root_element.getFirstElement("XMLScript"):
            self.cc3dSimulationData.xmlScript = root_element.getFirstElement("XMLScript").getText()
#            print MODULENAME,'  readCC3DFileFormat():  found XMLScript, getText=',self.cc3dSimulationData.xmlScript
            self.cc3dSimulationData.xmlScript = os.path.abspath(os.path.join(bp,self.cc3dSimulationData.xmlScript)) #normalizing path to xml script
            self.cc3dSimulationData.xmlScriptResource.path = self.cc3dSimulationData.xmlScript
            self.cc3dSimulationData.xmlScriptResource.type = "XMLScript"
            
        if root_element.getFirstElement("PythonScript"):
            self.cc3dSimulationData.pythonScript = root_element.getFirstElement("PythonScript").getText()
#            print MODULENAME,'  readCC3DFileFormat():  found PythonScript, getText=',self.cc3dSimulationData.pythonScript
            self.cc3dSimulationData.pythonScript = os.path.abspath(os.path.join(bp,self.cc3dSimulationData.pythonScript)) #normalizing path to python script
            self.cc3dSimulationData.pythonScriptResource.path = self.cc3dSimulationData.pythonScript
            self.cc3dSimulationData.pythonScriptResource.type = "PythonScript"

        if root_element.getFirstElement("PIFFile"):
            self.cc3dSimulationData.pifFile = root_element.getFirstElement("PIFFile").getText()
#            print MODULENAME,'  readCC3DFileFormat():  found PIFFile'
            self.cc3dSimulationData.pifFile = os.path.abspath(os.path.join(bp,self.cc3dSimulationData.pifFile)) #normalizing path
            self.cc3dSimulationData.pifFileResource.path = self.cc3dSimulationData.pifFile
            self.cc3dSimulationData.pifFileResource.type = "PIFFile"
            
        if root_element.getFirstElement("WindowScript"):
            self.cc3dSimulationData.windowScript = root_element.getFirstElement("WindowScript").getText()
#            print MODULENAME,'  readCC3DFileFormat():  found WindowScript, getText=',self.cc3dSimulationData.windowScript
            self.cc3dSimulationData.windowScript = os.path.abspath(os.path.join(bp,self.cc3dSimulationData.windowScript)) #normalizing path
            self.cc3dSimulationData.windowScriptResource.path = self.cc3dSimulationData.windowScript
            self.cc3dSimulationData.windowScriptResource.type = "WindowScript"
#                    if playerSettingsElement:
#            windowsXml2Obj = XMLUtils.Xml2Obj()
            winXml2ObjConverter = XMLUtils.Xml2Obj()
            winRoot = winXml2ObjConverter.Parse(self.cc3dSimulationData.windowScript)
#            print MODULENAME,'  readCC3DFileFormat():  type(winRoot)=',type(winRoot)
#            winList = winRoot.getElements("Window")

            winList = XMLUtils.CC3DXMLListPy(winRoot.getElements("Window"))
#            print MODULENAME,'  readCC3DFileFormat():  type(winList)=',type(winList)
            #  The following is pretty ugly; there's probably a more elegant way to parse this, but this works 
            for myWin in winList:
#                print MODULENAME,'  readCC3DFileFormat():  dir(myWin)=',dir(myWin)
#                print MODULENAME,'  readCC3DFileFormat():  myWin.getAttributes()=',myWin.getAttributes()
#                print MODULENAME,'  readCC3DFileFormat():  Element=',myWin.name
                attrKeys = myWin.getAttributes().keys()
#                print MODULENAME,'  readCC3DFileFormat():  attrKeys=',attrKeys
                winName = myWin.getAttribute("Name")
#                print MODULENAME,'  readCC3DFileFormat():  got Name= ',winName
                locElms = myWin.getElements("Location")
                elms = XMLUtils.CC3DXMLListPy(locElms)
                for elm in elms:
                    xpos = elm.getAttributeAsInt("x")
                    ypos = elm.getAttributeAsInt("y")
#                    print MODULENAME,'  readCC3DFileFormat():  Location x,y= ',xpos,ypos
                
                sizeElms = myWin.getElements("Size")
                elms = XMLUtils.CC3DXMLListPy(sizeElms)
                for elm in elms:
                    width = elm.getAttributeAsInt("width")
                    height = elm.getAttributeAsInt("height")
#                    print MODULENAME,'  readCC3DFileFormat():  Size width,height= ',width,height
                    
                self.cc3dSimulationData.windowDict[winName] = [xpos,ypos,width,height]
                
            print MODULENAME,'  -------- self.cc3dSimulationData.windowDict= ',self.cc3dSimulationData.windowDict
                

                
                
        if root_element.getFirstElement("SerializeSimulation"):
            serializeElem = root_element.getFirstElement("SerializeSimulation")
            self.cc3dSimulationData.serializerResource = CC3DSerializerResource()
            if serializeElem:
                if serializeElem.findAttribute("OutputFrequency"):
                    self.cc3dSimulationData.serializerResource.outputFrequency = serializeElem.getAttributeAsInt("OutputFrequency")
                    
                if serializeElem.findAttribute("AllowMultipleRestartDirectories"):                        
                    self.cc3dSimulationData.serializerResource.allowMultipleRestartDirectories = serializeElem.getAttributeAsBool("AllowMultipleRestartDirectories")

                if serializeElem.findAttribute("FileFormat"):
                    self.cc3dSimulationData.serializerResource.fileFormat = serializeElem.getAttribute("FileFormat")
                    
        if root_element.getFirstElement("RestartSimulation"):
            restartElem = root_element.getFirstElement("RestartSimulation")
            if not self.cc3dSimulationData.serializerResource:
                self.cc3dSimulationData.serializerResource = CC3DSerializerResource()
            
            if restartElem.findAttribute("RestartDirectory"):                        
                self.cc3dSimulationData.serializerResource.restartDirectory = restartElem.getAttribute("RestartDirectory")
        

        if root_element.getFirstElement("ParameterScan"):
            psFile = root_element.getFirstElement("ParameterScan").getText()
            self.cc3dSimulationData.parameterScanResource= CC3DParameterScanResource()
            self.cc3dSimulationData.parameterScanResource.path=os.path.abspath(os.path.join(bp,psFile)) #normalizing path to python script
            self.cc3dSimulationData.parameterScanResource.type = 'ParameterScan'
            self.cc3dSimulationData.parameterScanResource.basePath=self.cc3dSimulationData.basePath # setting same base path for parameter scan as for the project - necessary to get relative paths in the parameterSpec file
            #reading content of XML parameter scan specs
            # ------------------------------------------------------------------ IMPORTANT IMPOTRANT ------------------------------------------------------------------
            # WE HAVE TO CALL MANUALLYreadParameterScanSpecs because if it is called each time CC3DSiulationDataHandler calls readCC3DFileFormat it may cause problems with parameter scan
            # namely one process will attempt to read parameter scan specs while another might try to write to it and error will get thrown and synchronization gets lost
            # plus readCC3DFileFormat should read .cc3d only , not files which are included from .cc3d
            # ------------------------------------------------------------------ IMPORTANT IMPOTRANT ------------------------------------------------------------------            
            # # # self.cc3dSimulationData.parameterScanResource.readParameterScanSpecs()
        
        resourceList = XMLUtils.CC3DXMLListPy(root_element.getElements("Resource"))
#        print MODULENAME,'  readCC3DFileFormat():  resourceList=',resourceList
        for resourceElem in resourceList:
            cc3dResource = CC3DResource()
            cc3dResource.path = os.path.abspath(os.path.join(bp,resourceElem.getText()))
            
            if resourceElem.findAttribute("Type"):
                cc3dResource.type = resourceElem.getAttribute("Type")
#                print MODULENAME,'  readCC3DFileFormat():  cc3dResource.type=',cc3dResource.type  # e.g. "Python" or "XML"
            if resourceElem.findAttribute("Module"):
                cc3dResource.module = resourceElem.getAttribute("Module")
            if resourceElem.findAttribute("Origin"):
                cc3dResource.origin = resourceElem.getAttribute("Origin")
            if resourceElem.findAttribute("Copy"):
                copyAttr = resourceElem.getAttribute("Copy")
                if copyAttr.lower()=="no":
                    cc3dResource.copy=False

            self.cc3dSimulationData.resources[cc3dResource.path] = cc3dResource
#        print MODULENAME,'  readCC3DFileFormat():  self.cc3dSimulationData.resources=',self.cc3dSimulationData.resources
#        print MODULENAME,'  readCC3DFileFormat():  --> .resources.keys()=',self.cc3dSimulationData.resources.keys()            
    def formatResourceElement(self,_resource,_elementName=""):
        elName=""
        if _elementName!="":
            elName=_elementName
        else:
            elName="Resource"
        
        attributeDict={}
        if _resource.type!="":
            attributeDict["Type"]=_resource.type
            
        if _resource.module!="":
            attributeDict["Module"]=_resource.module
        
        if _resource.origin!="":
            attributeDict["Origin"]=_resource.origin
            
        if not _resource.copy:
            attributeDict["Copy"]="No"
            
        return elName,attributeDict,findRelativePath(self.cc3dSimulationData.basePath,_resource.path)
    
    def writeCC3DFileFormat(self,_fileName):        
        from XMLUtils import ElementCC3D
        csd=self.cc3dSimulationData
        simulationElement = ElementCC3D("Simulation",{"version":csd.version})
        
        if csd.xmlScriptResource.path!="":
            elName,attributeDict,path = self.formatResourceElement(csd.xmlScriptResource,"XMLScript")
            simulationElement.ElementCC3D(elName,attributeDict,path)
            
        if csd.pythonScriptResource.path!="":
            elName,attributeDict,path = self.formatResourceElement(csd.pythonScriptResource,"PythonScript")
            simulationElement.ElementCC3D(elName,attributeDict,path)
            
        if csd.pifFileResource.path!="":
            elName,attributeDict,path = self.formatResourceElement(csd.pifFileResource,"PIFFile")
            simulationElement.ElementCC3D(elName,attributeDict,path)
            
        if csd.windowScriptResource.path!="":
            elName,attributeDict,path = self.formatResourceElement(csd.windowScriptResource,"WindowScript")
            simulationElement.ElementCC3D(elName,attributeDict,path)
            
        
        resourcesDict={}
        #storing resources in a dictionary using resource type as a key   
        for resourceKey, resource in csd.resources.iteritems():
            if resource.type=="PIFFile" and csd.pifFileResource.path==resource.path:
                print MODULENAME,"IGNORING RESOURCE =",resource.path
                continue
            
            try:
                resourcesDict[resource.type].append(resource)
            except LookupError,e:
                resourcesDict[resource.type]=[resource]
                
            # elName,attributeDict,path = self.formatResourceElement(resource)
            # simulationElement.ElementCC3D(elName,attributeDict,path)

        #sort resources according to path name
        for resourceType,resourceList in resourcesDict.iteritems():
            resourceList = sorted(resourceList, key=lambda x: x.path)
            
            # after sorting have to reinsert list into dictionary to have it available later
            resourcesDict[resourceType]=resourceList

        sortedResourceTypeNames=resourcesDict.keys()
        sortedResourceTypeNames.sort()
        
        for resourceType in sortedResourceTypeNames:
            for resource in resourcesDict[resourceType]:
                elName,attributeDict,path = self.formatResourceElement(resource)
                simulationElement.ElementCC3D(elName,attributeDict,path)
                
        if csd.serializerResource:
            csd.serializerResource.appendXMLStub(simulationElement)
            
        if csd.parameterScanResource:
            elName,attributeDict,path = self.formatResourceElement(csd.parameterScanResource,'ParameterScan')
            simulationElement.ElementCC3D(elName,attributeDict,path)
            
        simulationElement.CC3DXMLElement.saveXML(str(_fileName))   
    
    # Based on code by  Cimarron Taylor
    # Date: July 6, 2003
    

    