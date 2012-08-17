# -*- coding: utf-8 -*-
import os, sys
import string

import SimpleTabView

from  Graphics.GraphicsFrameWidget import GraphicsFrameWidget

MODULENAME = '---- ScreenshotManager.py: '

class ScreenshotData:
    def __init__(self):
        self.screenshotName=""
        self.screenshotCoreName=""
        self.spaceDimension="2D"
        self.projection="xy"
        self.plotData=("Cell_Field","CellField") # this is a tupple there first element is field name (as displayed in the field list in the player) and the second one is plot type (e.g. CellField, Confield, Vector Field)
        self.projectionPosition=0
        self.screenshotGraphicsWidget=None
        # self.originalCameraObj=None
        #those are the values used to handle 3D screenshots
        
        self.clippingRange=None
        self.focalPoint=None
        self.position=None
        self.viewUp=None
        
    def extractCameraInfo(self,_camera):
        self.clippingRange=_camera.GetClippingRange()
        self.focalPoint=_camera.GetFocalPoint()
        self.position=_camera.GetPosition()
        self.viewUp=_camera.GetViewUp()
        
        
    def extractCameraInfoFromList(self,_cameraSettings):
        self.clippingRange=_cameraSettings[0:2]
        self.focalPoint=_cameraSettings[2:5]
        self.position=_cameraSettings[5:8]
        self.viewUp=_cameraSettings[8:11]
        print MODULENAME,"self.clippingRange=",self.clippingRange
        print "self.focalPoint",self.focalPoint
        print "self.position=",self.position
        print "self.viewUp=",self.viewUp
        print "camerasettings=",_cameraSettings
        
    def prepareCamera(self):
        if self.clippingRange and self.focalPoint and self.position and self.viewUp:         
            cam=self.screenshotGraphicsWidget.getCamera()
            cam.SetClippingRange(self.clippingRange)
            cam.SetFocalPoint(self.focalPoint)
            cam.SetPosition(self.position)
            cam.SetViewUp(self.viewUp)

    def compareCameras(self,_camera):
        _clippingRange=_camera.GetClippingRange()
        if self.clippingRange != _camera.GetClippingRange():
            return False
        if self.focalPoint!=_camera.GetFocalPoint():
            return False
        if self.position!=_camera.GetPosition():
            return False
        if self.viewUp!=_camera.GetViewUp():
            return False
        return True
    def compareExistingCameraToNewCameraSettings(self, _cameraSettings):
        if self.clippingRange[0]!=_cameraSettings[0] or self.clippingRange[1]!=_cameraSettings[1]:
            return False
        if self.focalPoint[0]!=_cameraSettings[2] or self.focalPointe[1]!=_cameraSettings[3] or self.focalPoint[3]!=_cameraSettings[4]:
            return False
        if self.position[0]!=_cameraSettings[5] or self.position[1]!=_cameraSettings[6] or self.position[3]!=_cameraSettings[7]:
            return False
        if self.viewUp[0]!=_cameraSettings[8] or self.viewUp[1]!=_cameraSettings[9] or self.viewUp[3]!=_cameraSettings[10]:
            return False

            
class ScreenshotManager:
    def __init__(self,_tabViewWidget):
        self.screenshotDataDict={}
        self.tabViewWidget=_tabViewWidget
        print
#        print MODULENAME,'  ScreenshotManager: __init__(),   self.tabViewWidget=',self.tabViewWidget
#        print MODULENAME,'  ScreenshotManager: __init__(),   type(self.tabViewWidget)=',type(self.tabViewWidget)
#        print MODULENAME,'  ScreenshotManager: __init__(),   dir(self.tabViewWidget)=',dir(self.tabViewWidget)
        # self.sim=self.tabViewWidget.sim
        self.basicSimulationData=self.tabViewWidget.basicSimulationData
        self.screenshotNumberOfDigits=len(str(self.basicSimulationData.numberOfSteps))
        # self.screenshotNumberOfDigits=len(str(self.sim.getNumSteps()))
        self.maxNumberOfScreenshots=20 # we limit max number of screenshots to discourage users from using screenshots as their main analysis tool
                                       #a better solution is to store latice to a pif file and then do postprocessing 
        self.screenshotCounter3D=0
        
        self.screenshotGraphicsWidget = GraphicsFrameWidget(self.tabViewWidget)
#        print MODULENAME,'  ScreenshotManager: __init__(),   self.screenshotGraphicsWidget=',self.screenshotGraphicsWidget
#        print MODULENAME,'  ScreenshotManager: __init__(),   self.screenshotGraphicsWidget.winId().__int__()=',self.screenshotGraphicsWidget.winId().__int__()
#        print
#        import pdb; pdb.set_trace()
#        bad = 1/0
#        SimpleTabView.   # rwh: add this to the graphics windows dict

#        self.tabViewWidget.lastActiveWindow = self.screenshotGraphicsWidget
#        self.tabViewWidget.updateActiveWindowVisFlags()
        
        self.screenshotGraphicsWidget.readSettings()
        self.tabViewWidget.addSubWindow(self.screenshotGraphicsWidget)
        self.screenshotGraphicsWidgetFieldTypesInitialized=False
        
        
        
    def produceScreenshotCoreName(self,_scrData):
        return str(_scrData.plotData[0])+"_"+str(_scrData.plotData[1])
        
    def produceScreenshotName(self,_scrData):
        screenshotName="Screenshot"
        screenshotCoreName="Screenshot"
        
        if _scrData.spaceDimension=="2D":
            screenshotCoreName=self.produceScreenshotCoreName(_scrData)
            screenshotName=screenshotCoreName+"_"+_scrData.spaceDimension+"_"+_scrData.projection+"_"+str(_scrData.projectionPosition)
        elif _scrData.spaceDimension=="3D":
            screenshotCoreName=self.produceScreenshotCoreName(_scrData)
            screenshotName=screenshotCoreName+"_"+_scrData.spaceDimension+"_"+str(self.screenshotCounter3D)
        return (screenshotName,screenshotCoreName)
    
    def writeScreenshotDescriptionFile(self,_fileName):
        from XMLUtils import ElementCC3D
        
        
        screenshotFileElement=ElementCC3D("CompuCell3DScreenshots")
        
        for name in self.screenshotDataDict:
            scrData=self.screenshotDataDict[name]
            if scrData.spaceDimension=="2D":
                scrDescElement=screenshotFileElement.ElementCC3D("ScreenshotDescription")
                
                scrDescElement.ElementCC3D("Dimension", {}, str(scrData.spaceDimension))
                
                scrDescElement.ElementCC3D("Plot", {"PlotType":str(scrData.plotData[1]),"PlotName":str(scrData.plotData[0])})
                
                scrDescElement.ElementCC3D("Projection", {"ProjectionPlane":scrData.projection,"ProjectionPosition":str(scrData.projectionPosition)})
                
                scrDescElement.ElementCC3D("Size", {"Width":str(scrData.screenshotGraphicsWidget.size().width()),"Height":str(scrData.screenshotGraphicsWidget.size().height())})
                
            if scrData.spaceDimension=="3D":
                scrDescElement=screenshotFileElement.ElementCC3D("ScreenshotDescription")                
                
                scrDescElement.ElementCC3D("Dimension", {}, str(scrData.spaceDimension))
                
                scrDescElement.ElementCC3D("Plot", {"PlotType":str(scrData.plotData[1]),"PlotName":str(scrData.plotData[0])})
                
                scrDescElement.ElementCC3D("CameraClippingRange", {"Min":str(scrData.clippingRange[0]),"Max":str(scrData.clippingRange[1])})
                
                scrDescElement.ElementCC3D("CameraFocalPoint", {"x":str(scrData.focalPoint[0]),"y":str(scrData.focalPoint[1]),"z":str(scrData.focalPoint[2])})
                
                scrDescElement.ElementCC3D("CameraPosition", {"x":str(scrData.position[0]),"y":str(scrData.position[1]),"z":str(scrData.position[2])})
                
                scrDescElement.ElementCC3D("CameraViewUp", {"x":str(scrData.viewUp[0]),"y":str(scrData.viewUp[1]),"z":str(scrData.viewUp[2])})
                
                scrDescElement.ElementCC3D("Size", {"Width":str(scrData.screenshotGraphicsWidget.size().width()),"Height":str(scrData.screenshotGraphicsWidget.size().height())})
                
        
        screenshotFileElement.CC3DXMLElement.saveXML(str(_fileName))
        
    def readScreenshotDescriptionFile(self,_fileName):        
        import XMLUtils
        
        xml2ObjConverter = XMLUtils.Xml2Obj()
        root_element=xml2ObjConverter.Parse(_fileName)
        scrList=XMLUtils.CC3DXMLListPy(root_element.getElements("ScreenshotDescription"))
        for scr in scrList:
            if scr.getFirstElement("Dimension").getText()=="2D":
                print MODULENAME,"GOT 2D SCREENSHOT"
                scrData=ScreenshotData()
                scrData.spaceDimension="2D"
                
                plotElement=scr.getFirstElement("Plot")
                scrData.plotData=(plotElement.getAttribute("PlotName"),plotElement.getAttribute("PlotType"))
                
                projElement=scr.getFirstElement("Projection")
                scrData.projection=projElement.getAttribute("ProjectionPlane")
                scrData.projectionPosition=int(projElement.getAttribute("ProjectionPosition"))
                
                sizeElement=scr.getFirstElement("Size")
                scrSize=[int(sizeElement.getAttribute("Width")),int(sizeElement.getAttribute("Height"))]
                
                # scrData initialized now will initialize graphics widget
                (scrName,scrCoreName)=self.produceScreenshotName(scrData)
                if not scrName in self.screenshotDataDict:
                    scrData.screenshotName=scrName
                    scrData.screenshotCoreName=scrCoreName

                    scrData.screenshotGraphicsWidget=self.screenshotGraphicsWidget

                    self.screenshotDataDict[scrData.screenshotName]=scrData
                else:
                    print MODULENAME,"Screenshot ",scrName," already exists"
                
            elif scr.getFirstElement("Dimension").getText()=="3D":
                scrData=ScreenshotData()
                scrData.spaceDimension="3D"
                
                plotElement=scr.getFirstElement("Plot")
                scrData.plotData=(plotElement.getAttribute("PlotName"),plotElement.getAttribute("PlotType"))
                
                sizeElement=scr.getFirstElement("Size")
                scrSize=[int(sizeElement.getAttribute("Width")),int(sizeElement.getAttribute("Height"))]
                
                (scrName,scrCoreName)=self.produceScreenshotName(scrData)
                print MODULENAME,"(scrName,scrCoreName)=",(scrName,scrCoreName)
                okToAddScreenshot=True
                
                # extracting Camera Settings
                camSettings=[]
                
                clippingRangeElement=scr.getFirstElement("CameraClippingRange")
                camSettings.append(float(clippingRangeElement.getAttribute("Min")))
                camSettings.append(float(clippingRangeElement.getAttribute("Max")))
                
                focalPointElement=scr.getFirstElement("CameraFocalPoint")
                camSettings.append(float(focalPointElement.getAttribute("x")))
                camSettings.append(float(focalPointElement.getAttribute("y")))
                camSettings.append(float(focalPointElement.getAttribute("z")))
                
                positionElement=scr.getFirstElement("CameraPosition")
                camSettings.append(float(positionElement.getAttribute("x")))
                camSettings.append(float(positionElement.getAttribute("y")))
                camSettings.append(float(positionElement.getAttribute("z")))
                
                viewUpElement=scr.getFirstElement("CameraViewUp")
                camSettings.append(float(viewUpElement.getAttribute("x")))
                camSettings.append(float(viewUpElement.getAttribute("y")))
                camSettings.append(float(viewUpElement.getAttribute("z")))
                
                for name in self.screenshotDataDict:
                    scrDataFromDict=self.screenshotDataDict[name]
                    if scrDataFromDict.screenshotCoreName==scrCoreName and scrDataFromDict.spaceDimension=="3D":
                        print MODULENAME,"scrDataFromDict.screenshotCoreName=",scrDataFromDict.screenshotCoreName," scrCoreName=",scrCoreName
                        
                        if scrDataFromDict.compareExistingCameraToNewCameraSettings(camSettings):
                            print MODULENAME,"CAMERAS ARE THE SAME"
                            okToAddScreenshot=False
                            break
                        else:
                            print MODULENAME,"CAMERAS ARE DIFFERENT"
                print MODULENAME,"okToAddScreenshot=",okToAddScreenshot    
                
                if (not scrName in self.screenshotDataDict) and okToAddScreenshot:
                    scrData.screenshotName=scrName
                    scrData.screenshotCoreName=scrCoreName            
                    
                    scrData.screenshotGraphicsWidget=self.screenshotGraphicsWidget
                    
                    scrData.extractCameraInfoFromList(camSettings)
                    self.screenshotDataDict[scrData.screenshotName]=scrData
                
            else:
                print MODULENAME,"GOT UNKNOWN SCREENSHOT"
            
           
    def add2DScreenshot(self, _plotName,_plotType,_projection,_projectionPosition):
        if len(self.screenshotDataDict)>self.maxNumberOfScreenshots:
            print MODULENAME,"MAX NUMBER OF SCREENSHOTS HAS BEEN REACHED"
        
        scrData=ScreenshotData()
        scrData.spaceDimension="2D"
        scrData.plotData=(_plotName,_plotType)        
        print MODULENAME," add2DScreenshot(): scrData.plotData=",scrData.plotData   # e.g. =('Cell_Field', 'CellField')
        scrData.projection=_projection
        scrData.projectionPosition=int(_projectionPosition)
        
        (scrName,scrCoreName)=self.produceScreenshotName(scrData)
        
        print MODULENAME,"  add2DScreenshot():  THIS IS NEW SCRSHOT NAME",scrName   # e.g. Cell_Field_CellField_2D_XY_150
        if not scrName in self.screenshotDataDict:
            scrData.screenshotName=scrName
            scrData.screenshotCoreName=scrCoreName
            scrData.screenshotGraphicsWidget=self.screenshotGraphicsWidget
#            print MODULENAME," add2DScreenshot(): scrData.screenshotGraphicsWidget=",scrData.screenshotGraphicsWidget
#            print MODULENAME," add2DScreenshot(): type(scrData.screenshotGraphicsWidget)=",type(scrData.screenshotGraphicsWidget)
            
#            self.tabViewWidget.lastActiveWindow = self.screenshotGraphicsWidget
            print MODULENAME," add2DScreenshot(): win id=", self.tabViewWidget.lastActiveWindow.winId().__int__()
            self.tabViewWidget.updateActiveWindowVisFlags(self.screenshotGraphicsWidget)
            
            # on linux there is a problem with X-server/Qt/QVTK implementation and calling resize right after additional QVTK 
            # is created causes segfault so possible "solution" is to do resize right before taking screenshot. It causes flicker but does not cause segfault
            if sys.platform=='Linux' or sys.platform=='linux' or sys.platform=='linux2': 
                pass
            else:
                self.screenshotDataDict[scrData.screenshotName]=scrData
        else:
            print MODULENAME,"Screenshot ",scrName," already exists"


    def add3DScreenshot(self, _plotName,_plotType, _camera):
        if len(self.screenshotDataDict)>self.maxNumberOfScreenshots:
            print MODULENAME,"MAX NUMBER OF SCREENSHOTS HAS BEEN REACHED"        
        scrData=ScreenshotData()
        scrData.spaceDimension="3D"
        scrData.plotData=(_plotName,_plotType)
        print MODULENAME,"add3DScreenshot(): scrData.plotData",scrData.plotData
        (scrName,scrCoreName)=self.produceScreenshotName(scrData)

        print MODULENAME,"add3DScreenshot(): scrName",scrName
        okToAddScreenshot=True
        for name in self.screenshotDataDict:
            scrDataFromDict=self.screenshotDataDict[name]
            if scrDataFromDict.screenshotCoreName==scrCoreName and scrDataFromDict.spaceDimension=="3D":
                if scrDataFromDict.compareCameras(_camera):
                    print MODULENAME,"CAMERAS ARE THE SAME"
                    okToAddScreenshot=False
                    break
                else:
                    print MODULENAME,"CAMERAS ARE DIFFERENT"
                    
        print MODULENAME,"okToAddScreenshot=",okToAddScreenshot    
        if (not scrName in self.screenshotDataDict) and okToAddScreenshot:
            scrData.screenshotName=scrName
            scrData.screenshotCoreName=scrCoreName
            scrData.screenshotGraphicsWidget=self.screenshotGraphicsWidget 
#            self.tabViewWidget.lastActiveWindow = self.screenshotGraphicsWidget
            print MODULENAME," add3DScreenshot(): win id=", self.tabViewWidget.lastActiveWindow.winId().__int__()
            self.tabViewWidget.updateActiveWindowVisFlags(self.screenshotGraphicsWidget)
            
            scrData.extractCameraInfo(_camera)

            # on linux there is a problem with X-server/Qt/QVTK implementation and calling resize right after additional QVTK 
            # is created causes segfault so possible "solution" is to do resize right before taking screenshot. It causes flicker but does not cause segfault
            if sys.platform=='Linux' or sys.platform=='linux' or sys.platform=='linux2':
                pass
            else:
                self.screenshotDataDict[scrData.screenshotName] = scrData
                self.screenshotCounter3D += 1
        else:
            print MODULENAME,"Screenshot ",scrCoreName," with current camera settings already exists. You need to rotate camera i.e. rotate picture using moouse to take additional screenshot"
            
            
    def outputScreenshots(self,_generalScreenshotDirectoryName,_mcs):  # called from SimpleTabView:handleCompletedStep{Regular,CML*}
        print MODULENAME, 'outputScreenshots():  _generalScreenshotDirectoryName=',_generalScreenshotDirectoryName
        mcsFormattedNumber = string.zfill(str(_mcs),self.screenshotNumberOfDigits) # fills string with 0's up to self.screenshotNumberOfDigits width
        
        if not self.screenshotGraphicsWidgetFieldTypesInitialized:
            self.screenshotGraphicsWidget.setFieldTypesComboBox(self.tabViewWidget.fieldTypes)
        
        # self.screenshotGraphicsWidget.setShown(True)
        self.screenshotGraphicsWidget.qvtkWidget.resize(400,400)   # rwh, why??
#        self.screenshotGraphicsWidget.qvtkWidget.resize(343,300)   # rwh, why??

#        print MODULENAME,'outputScreenshots(): win size=',  self.tabViewWidget.mainGraphicsWindow.size() # PyQt4.QtCore.QSize(367, 378)
        
        for scrName in self.screenshotDataDict.keys():
            print MODULENAME,'outputScreenshots(): scrName =',scrName  # e.g. FGF_ConField_2D_XY_0  (i.e. subdir name)
            scrData = self.screenshotDataDict[scrName]
            scrFullDirName = os.path.join(_generalScreenshotDirectoryName,scrData.screenshotName)
            
            if not os.path.isdir(scrFullDirName):# will create screenshot directory if directory does not exist
                os.mkdir(scrFullDirName)
            scrFullName = os.path.join(scrFullDirName,scrData.screenshotName+"_"+mcsFormattedNumber+".png")
            
#            print MODULENAME,'outputScreenshots(): scrData.spaceDimension =',scrData.spaceDimension  # 2D

            # rwh: why is this necessary??
            if scrData.spaceDimension=="2D":  # cf. this block with how it's done in MVCDrawView2D.py:takeSimShot()
                self.screenshotGraphicsWidget.setDrawingStyle("2D")
                self.screenshotGraphicsWidget.draw2D.initSimArea(self.basicSimulationData)
                self.screenshotGraphicsWidget.draw2D.setPlane(scrData.projection,scrData.projectionPosition)
                scrData.prepareCamera()
                
            elif scrData.spaceDimension=="3D":
                self.screenshotGraphicsWidget.setDrawingStyle("3D")
                self.screenshotGraphicsWidget.draw3D.initSimArea(self.basicSimulationData)
                scrData.prepareCamera()
                
            else:
                print MODULENAME,' WARNING - got to unexpected return'
                return # should not get here
            
            # #have to set up camera
            # scrData.prepareCamera()
            
                        
            # self.screenshotGraphicsWidget.drawField(self.basicSimulationData,scrData.plotData)
            # self.screenshotGraphicsWidget.setFieldTypes(scrData.plotData)
#            print MODULENAME,"   before drawFieldLocal, scrData.plotData=",scrData.plotData
            self.screenshotGraphicsWidget.setPlotData(scrData.plotData)
            self.screenshotGraphicsWidget.drawFieldLocal(self.basicSimulationData,False) # second argument tells drawFieldLocal fcn not to use combo box to get field name
#            print MODULENAME,"AFTER drawFieldLocal"
            
            # scrData.screenshotGraphicsWidget.setShown(True)
            # # scrData.screenshotGraphicsWidget.resize(400,400)

            # on linux there is a problem with X-server/Qt/QVTK implementation and calling resize right after additional QVTK 
            # is created causes segfault so possible "solution" is to do resize right before taking screenshot. It causes flicker but does not cause segfault
#            print MODULENAME,' sys.platform = ',sys.platform
            if sys.platform=='darwin' or sys.platform=='Linux' or sys.platform=='linux' or sys.platform=='linux2':
                scrData.screenshotGraphicsWidget.setShown(True)
#                scrData.screenshotGraphicsWidget.resize(self.tabViewWidget.mainGraphicsWindow.size())

            # scrData.screenshotGraphicsWidget.takeSimShot(scrFullName)
#            print MODULENAME, '  calling self.screenshotGraphicsWidget.takeSimShot(',scrFullName
            self.screenshotGraphicsWidget.takeSimShot(scrFullName)
                # scrData.screenshotGraphicsWidget.setShown(False)
        
            if sys.platform=='darwin' or sys.platform=='Linux' or sys.platform=='linux' or sys.platform=='linux2':
                scrData.screenshotGraphicsWidget.setShown(False)
