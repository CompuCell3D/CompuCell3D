# -*- coding: utf-8 -*-
import os, sys
import string

import Configuration
import SimpleTabView

from  Graphics.GraphicsFrameWidget import GraphicsFrameWidget

MODULENAME = '---- ScreenshotManager.py: '

class ScreenshotData:
    def __init__(self):
        self.screenshotName=""
        self.screenshotCoreName=""
        self.spaceDimension="2D"
        self.projection="xy"
        self.plotData=("Cell_Field","CellField") # this is a tuple where first element is field name (as displayed in the field list in the player5) and the second one is plot type (e.g. CellField, Confield, Vector Field)
        self.projectionPosition=0
        self.screenshotGraphicsWidget=None
        # self.originalCameraObj=None
        #those are the values used to handle 3D screenshots
        
        self.clippingRange=None
        self.focalPoint=None
        self.position=None
        self.viewUp=None
        
#        self.winWidth=299   # some unique default 
#        self.winHeight=299
        
    def extractCameraInfo(self,_camera):
        self.clippingRange=_camera.GetClippingRange()
        self.focalPoint=_camera.GetFocalPoint()
        self.position=_camera.GetPosition()
        self.viewUp=_camera.GetViewUp()
#        print MODULENAME,"-------- extractCameraInfoFromList(): Range,FP,Pos,Up=",self.clippingRange,self.focalPoint,self.position,self.viewUp
        
        
    def extractCameraInfoFromList(self,_cameraSettings):
        self.clippingRange=_cameraSettings[0:2]
        self.focalPoint=_cameraSettings[2:5]
        self.position=_cameraSettings[5:8]
        self.viewUp=_cameraSettings[8:11]
#        print MODULENAME,"-------- extractCameraInfoFromList():  self.clippingRange=",self.clippingRange
#        print "self.focalPoint",self.focalPoint
#        print "self.position=",self.position
#        print "self.viewUp=",self.viewUp
#        print "camerasettings=",_cameraSettings
        
    def prepareCamera(self):
#        print MODULENAME,"------------  prepareCamera()"
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
        from weakref import ref
        self.tabViewWidget=ref(_tabViewWidget)
        tvw=self.tabViewWidget()
        # print
#        print MODULENAME,'  ScreenshotManager: __init__(),   self.tabViewWidget=',self.tabViewWidget
#        print MODULENAME,'  ScreenshotManager: __init__(),   type(self.tabViewWidget)=',type(self.tabViewWidget)
#        print MODULENAME,'  ScreenshotManager: __init__(),   dir(self.tabViewWidget)=',dir(self.tabViewWidget)
        # self.sim=self.tabViewWidget.sim
        
        self.basicSimulationData=tvw.basicSimulationData
        self.basicSimulationData=tvw.basicSimulationData
        self.screenshotNumberOfDigits=len(str(self.basicSimulationData.numberOfSteps))
        
        # self.screenshotNumberOfDigits=len(str(self.sim.getNumSteps()))
        self.maxNumberOfScreenshots=20 # we limit max number of screenshots to discourage users from using screenshots as their main analysis tool
                                       #a better solution is to store latice to a pif file and then do postprocessing 
        self.screenshotCounter3D=0
        
        # self.screenshotGraphicsWidget = GraphicsFrameWidget(self.tabViewWidget)
        print 'CREATING SCREENSHOT WINDOW'

        self.screenshotGraphicsWidget = None
        # MDIFIX - temporary
        # return


        # MDIFIX
        self.screenshotGraphicsWidget = GraphicsFrameWidget(tvw, tvw)
        # self.screenshotGraphicsWidget.allowSaveLayout = False # we do not save screenshot widget in the windows layout

        # important because e.g. we do not save screenshot widget in the windows layout
        self.screenshotGraphicsWidget.is_screenshot_widget = True
        # self.screenshotGraphicsWidget = GraphicsFrameWidget(tvw)

        self.screenshotGraphicsWidget.screenshotWindowFlag = True
        
        xSize = Configuration.getSetting("Screenshot_X")
        ySize = Configuration.getSetting("Screenshot_Y")

        # xSize = 1000
        # ySize = 1000

        # print 'xSize=',xSize,' ySize=',ySize
        # self.screenshotGraphicsWidget.resize(xSize,ySize)

        self.screenshotGraphicsWidget.qvtkWidget.GetRenderWindow().SetSize(xSize,ySize) # default size
        self.screenshotGraphicsWidget.qvtkWidget.resize(xSize,ySize)
        
        winsize = self.screenshotGraphicsWidget.qvtkWidget.GetRenderWindow().GetSize()
        print 'ADDITIONAL SCREENSHOT WINDOW SIZE=',winsize
        
#        print MODULENAME,'  ScreenshotManager: __init__(),   self.screenshotGraphicsWidget=',self.screenshotGraphicsWidget
#        print MODULENAME,'  ScreenshotManager: __init__(),   self.screenshotGraphicsWidget.winId().__int__()=',self.screenshotGraphicsWidget.winId().__int__()
#        print
#        import pdb; pdb.set_trace()
#        bad = 1/0
#        SimpleTabView.   # rwh: add this to the graphics windows dict

#        self.tabViewWidget.lastActiveWindow = self.screenshotGraphicsWidget
#        self.tabViewWidget.updateActiveWindowVisFlags()
        
        self.screenshotGraphicsWidget.readSettings()
        # # # self.tabViewWidget.addSubWindow(self.screenshotGraphicsWidget)
        self.screenshotSubWindow = tvw.addSubWindow(self.screenshotGraphicsWidget)

        self.screenshotSubWindow.resize(xSize, ySize)

        
        #necessary to avoid spurious maximization of screenshot window. possible bug either in Player or in QMDIArea
        self.screenshotSubWindow.showMinimized()
        self.screenshotSubWindow.hide()
        
        
        self.screenshotGraphicsWidgetFieldTypesInitialized=False
        
        
    def cleanup(self):
        # have to do cleanup to ensure some of the memory intensive resources e.g. self.screenshotGraphicsWidget get deallocated
        if self.screenshotGraphicsWidget:
            print 'JUST BEFORE CLOSING self.screenshotGraphicsWidget'
            # this close and assignment do not do much for the non-mdi layout
            self.screenshotGraphicsWidget.close()
            self.screenshotGraphicsWidget=None
        self.tabViewWidget=None
        self.basicSimulationData=None
    
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
            
           
    def add2DScreenshot(self, _plotName,_plotType,_projection,_projectionPosition,_camera):   # called from GraphicsFrameWidget
        if len(self.screenshotDataDict)>self.maxNumberOfScreenshots:
            print MODULENAME,"MAX NUMBER OF SCREENSHOTS HAS BEEN REACHED"
        
        scrData=ScreenshotData()
        scrData.spaceDimension="2D"
        scrData.plotData=(_plotName,_plotType)        
        print MODULENAME," add2DScreenshot(): scrData.plotData=",scrData.plotData   # e.g. =('Cell_Field', 'CellField')
#        print MODULENAME," add2DScreenshot(): _camera=",_camera
#        print MODULENAME,"  add2DScreenshot():  incoming _camera: Range,FP,Pos,Up,Distance", \
#                _camera.GetClippingRange(),_camera.GetFocalPoint(),_camera.GetPosition(),_camera.GetViewUp(),_camera.GetDistance()
        scrData.projection = _projection
        scrData.projectionPosition = int(_projectionPosition)
        
#        import pdb; pdb.set_trace()
        
        (scrName,scrCoreName)=self.produceScreenshotName(scrData)
        
        print MODULENAME,"  add2DScreenshot():  THIS IS NEW SCRSHOT NAME",scrName   # e.g. Cell_Field_CellField_2D_XY_150
        
        if not scrName in self.screenshotDataDict:
            scrData.screenshotName = scrName
            scrData.screenshotCoreName = scrCoreName
            scrData.screenshotGraphicsWidget = self.screenshotGraphicsWidget   # = GraphicsFrameWidget (rf. __init__)
            
#            cam = self.screenshotGraphicsWidget.getCamera()
#            print MODULENAME,"  add2DScreenshot():  cam: Range,FP,Pos,Up,Distance", \
#                cam.GetClippingRange(),cam.GetFocalPoint(),cam.GetPosition(),cam.GetViewUp(),cam.GetDistance()
#            cam.SetClippingRange(_cam.GetClippingRange())
#            cam.SetDistance(_cam.GetDistance())
#            cam.SetFocalPoint(_cam.GetFocalPoint())
#            cam.SetPosition(_cam.GetPosition())
#            cam.SetThickness(_cam.GetThickness())
#            cam.SetViewUp(_cam.GetViewUp())   # not needed for 2D

#            print MODULENAME," add2DScreenshot(): scrData.screenshotGraphicsWidget=",scrData.screenshotGraphicsWidget
#            print MODULENAME," add2DScreenshot(): type(scrData.screenshotGraphicsWidget)=",type(scrData.screenshotGraphicsWidget)
            
#            self.tabViewWidget.lastActiveWindow = self.screenshotGraphicsWidget
#            print MODULENAME," add2DScreenshot(): win id=", self.tabViewWidget.lastActiveWindow.winId().__int__()
            tvw=self.tabViewWidget()
            if tvw:
                tvw.updateActiveWindowVisFlags(self.screenshotGraphicsWidget)
            
            scrData.extractCameraInfo(_camera)   # so "camera" icon (save images) remembers camera view
            
            # on linux there is a problem with X-server/Qt/QVTK implementation and calling resize right after additional QVTK 
            # is created causes segfault so possible "solution" is to do resize right before taking screenshot. 
            # It causes flicker but does not cause segfault.
            # User should NOT close or minimize this "empty" window (on Linux anyway).
            if sys.platform=='Linux' or sys.platform=='linux' or sys.platform=='linux2': 
#                pass
                self.screenshotDataDict[scrData.screenshotName] = scrData
            else:
                self.screenshotDataDict[scrData.screenshotName] = scrData
        else:
            print MODULENAME,"Screenshot ",scrName," already exists"


    def add3DScreenshot(self, _plotName,_plotType,_camera):   # called from GraphicsFrameWidget
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
                    
#        print MODULENAME,"okToAddScreenshot=",okToAddScreenshot    
        if (not scrName in self.screenshotDataDict) and okToAddScreenshot:
            scrData.screenshotName=scrName
            scrData.screenshotCoreName=scrCoreName
            scrData.screenshotGraphicsWidget=self.screenshotGraphicsWidget 
#            self.tabViewWidget.lastActiveWindow = self.screenshotGraphicsWidget
#            print MODULENAME," add3DScreenshot(): win id=", self.tabViewWidget.lastActiveWindow.winId().__int__()
            tvw=self.tabViewWidget()
            if tvw:
                tvw.updateActiveWindowVisFlags(self.screenshotGraphicsWidget)
            
            scrData.extractCameraInfo(_camera)

            # on linux there is a problem with X-server/Qt/QVTK implementation and calling resize right after additional QVTK 
            # is created causes segfault so possible "solution" is to do resize right before taking screenshot. 
            # It causes flicker but does not cause segfault
            # User should NOT close or minimize this "empty" window (on Linux anyway).
            if sys.platform=='Linux' or sys.platform=='linux' or sys.platform=='linux2':
#                pass
                self.screenshotDataDict[scrData.screenshotName] = scrData
                self.screenshotCounter3D += 1
            else:
                self.screenshotDataDict[scrData.screenshotName] = scrData
                self.screenshotCounter3D += 1
        else:
            print MODULENAME,"Screenshot ",scrCoreName," with current camera settings already exists. You need to rotate camera i.e. rotate picture using mouse to take additional screenshot"
            
            
    def outputScreenshots(self,_generalScreenshotDirectoryName,_mcs):  # called from SimpleTabView:handleCompletedStep{Regular,CML*}
#        print MODULENAME, 'outputScreenshots():  _generalScreenshotDirectoryName=',_generalScreenshotDirectoryName
        mcsFormattedNumber = string.zfill(str(_mcs),self.screenshotNumberOfDigits) # fills string with 0's up to self.screenshotNumberOfDigits width
        
        if not self.screenshotGraphicsWidgetFieldTypesInitialized:
            tvw=self.tabViewWidget()
            if tvw:
                self.screenshotGraphicsWidget.setFieldTypesComboBox(tvw.fieldTypes)
        
        # apparently on linux and most likely OSX we need to resize screenshot window before each screenshot
        xSize = Configuration.getSetting("Screenshot_X")
        ySize = Configuration.getSetting("Screenshot_Y")

        # xSize = 1000
        # ySize = 1000
        #
        # self.screenshotGraphicsWidget.resize(xSize,ySize)

        self.screenshotGraphicsWidget.qvtkWidget.GetRenderWindow().SetSize(xSize,ySize) # default size
        self.screenshotGraphicsWidget.qvtkWidget.resize(xSize,ySize)
        
        if not sys.platform.startswith('win'):   # we hide and restore screenshot window on linux and OSX only on windows it is not necessary
            self.screenshotSubWindow.showNormal()
            self.screenshotSubWindow.show()
        
        
# self.screenshotGraphicsWidget.setShown(True)
        winsize = self.screenshotGraphicsWidget.qvtkWidget.GetRenderWindow().GetSize()
        # print 'ADDITIONAL SCREENSHOT WINDOW SIZE=',winsize
# # #         self.screenshotGraphicsWidget.qvtkWidget.GetRenderWindow().SetSize(winsize[0],winsize[1])

#        print MODULENAME,'outputScreenshots(): type(winsize), [0],[1]=',type(winsize),winsize[0],winsize[1]
#        self.screenshotGraphicsWidget.qvtkWidget.resize(400,400)   # rwh, why?? (if I don't, it defaults to (100,30)
        # self.screenshotGraphicsWidget.qvtkWidget.resize(winsize[0],winsize[1])   # rwh, why?? (if I don't, it defaults to (100,30)        
        # self.screenshotGraphicsWidget.qvtkWidget.resize(611,411)   # rwh, why?? (if I don't, it defaults to (100,30)
#        print MODULENAME,'outputScreenshots(): dir(self.screenshotGraphicsWidget.qvtkWidget)=',dir(self.screenshotGraphicsWidget.qvtkWidget)
#        print MODULENAME,'outputScreenshots(): dir(self.screenshotGraphicsWidget.qvtkWidget.GetRenderWindow())=',dir(self.screenshotGraphicsWidget.qvtkWidget.GetRenderWindow())

#        print MODULENAME,'outputScreenshots(): self.screenshotGraphicsWidget.qvtkWidget.GetRenderWindow().GetSize()=',self.screenshotGraphicsWidget.qvtkWidget.GetRenderWindow().GetSize()
#        self.screenshotGraphicsWidget.qvtkWidget.resize(343,300)   # rwh, why??

#        print MODULENAME,'outputScreenshots(): win size=',  self.tabViewWidget.mainGraphicsWindow.size() # PyQt4.QtCore.QSize(367, 378)
        
#        print MODULENAME,'outputScreenshots(): self.screenshotDataDict=',self.screenshotDataDict
        for scrName in self.screenshotDataDict.keys():
#            print MODULENAME,'----------->>>>> outputScreenshots(): scrName =',scrName  # e.g. FGF_ConField_2D_XY_0  (i.e. subdir name)
            scrData = self.screenshotDataDict[scrName]
#            print MODULENAME,'outputScreenshots(): scrData.screenshotGraphicsWidget.size().width(), height() =',scrData.screenshotGraphicsWidget.size().width(),scrData.screenshotGraphicsWidget.size().height()  # 100,30 ?!
            scrFullDirName = os.path.join(_generalScreenshotDirectoryName,scrData.screenshotName)
            
            if not os.path.isdir(scrFullDirName):     # will create screenshot directory if directory does not exist
                print MODULENAME,'   outputScreenshots(): doing os.mkdir on scrFullDirName=',scrFullDirName
                os.mkdir(scrFullDirName)
                
            scrFullName = os.path.join(scrFullDirName,scrData.screenshotName+"_"+mcsFormattedNumber+".png")
            
#            print MODULENAME,'outputScreenshots(): scrData.spaceDimension =',scrData.spaceDimension  # 2D

            # rwh: why is this necessary??
            if scrData.spaceDimension=="2D":  # cf. this block with how it's done in MVCDrawView2D.py:takeSimShot()
                self.screenshotGraphicsWidget.setDrawingStyle("2D")
                self.screenshotGraphicsWidget.draw2D.initSimArea(self.basicSimulationData)   # MVCDrawViewBase
                self.screenshotGraphicsWidget.draw2D.setPlane(scrData.projection,scrData.projectionPosition)
                scrData.prepareCamera()
                
            elif scrData.spaceDimension=="3D":
                self.screenshotGraphicsWidget.setDrawingStyle("3D")
                self.screenshotGraphicsWidget.draw3D.initSimArea(self.basicSimulationData)
#                print MODULENAME,'outputScreenshots(): calling scrData.prepareCamera() '
                scrData.prepareCamera()
                
            else:
                print MODULENAME,' WARNING - got to unexpected return'
                return # should not get here
            
            # #have to set up camera
            # scrData.prepareCamera()
            
                        
            # self.screenshotGraphicsWidget.drawField(self.basicSimulationData,scrData.plotData)
            # self.screenshotGraphicsWidget.setFieldTypes(scrData.plotData)
#            print MODULENAME,"   before drawFieldLocal, scrData.plotData=",scrData.plotData

            print 'screenshot_widget=', self.screenshotGraphicsWidget.winId().__int__()


            self.screenshotGraphicsWidget.setPlotData(scrData.plotData)
            self.screenshotGraphicsWidget.drawFieldLocal(self.basicSimulationData,False) # second argument tells drawFieldLocal fcn not to use combo box to get field name
#            print MODULENAME,"AFTER drawFieldLocal"
            
            # scrData.screenshotGraphicsWidget.setShown(True)
            # # scrData.screenshotGraphicsWidget.resize(400,400)

            # on linux there is a problem with X-server/Qt/QVTK implementation and calling resize right after additional QVTK 
            # is created causes segfault so possible "solution" is to do resize right before taking screenshot. It causes flicker but does not cause segfault
#            print MODULENAME,' sys.platform = ',sys.platform
            if sys.platform == 'darwin' or sys.platform == 'Linux' or sys.platform == 'linux' or sys.platform == 'linux2':
                # scrData.screenshotGraphicsWidget.setShown(True)

                scrData.screenshotGraphicsWidget.show()
#                scrData.screenshotGraphicsWidget.resize(self.tabViewWidget.mainGraphicsWindow.size())

            # scrData.screenshotGraphicsWidget.takeSimShot(scrFullName)
            print MODULENAME, 'outputScreenshots():  calling self.screenshotGraphicsWidget.takeSimShot(',scrFullName
            self.screenshotGraphicsWidget.takeSimShot(scrFullName)
                # scrData.screenshotGraphicsWidget.setShown(False)
        
            if sys.platform == 'darwin' or sys.platform == 'Linux' or sys.platform == 'linux' or sys.platform == 'linux2':
                # scrData.screenshotGraphicsWidget.setShown(False)

                scrData.screenshotGraphicsWidget.hide()
                
        if not sys.platform.startswith('win'):      # we hide and restore screenshot window on linux and OSX only on windows it is not necessary
            self.screenshotSubWindow.showNormal()
            self.screenshotSubWindow.hide()
                
