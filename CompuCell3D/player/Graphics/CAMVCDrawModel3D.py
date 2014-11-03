import vtk 
import Configuration

VTK_MAJOR_VERSION=vtk.vtkVersion.GetVTKMajorVersion()

class CAMVCDrawModel3D(object):
    def __init__(self,_masterDrawModel=None):
        self.initMasterDrawModel(_masterDrawModel)
        self.mdm=None
        
    def initMasterDrawModel(self,_masterDrawModel=None):
        if _masterDrawModel is None:
            return
            
        from weakref import ref
        
        MDM = ref(_masterDrawModel)
        self.mdm=MDM()
                
    def initCellGlyphsActor3D(self,_glyphActor,_invisibleCellTypes):
        print 'INSIDE initCellGlyphsActor3D'
        fieldDim = self.mdm.currentDrawingParameters.bsd.fieldDim
        
        # # # import CA
        # # # pt=CA.Point3D()
        # # # ptVec=[0,0,0]        
        

        caManager = self.mdm.parentWidget.mysim()
        
        cellFieldS=caManager.getCellFieldS()

        centroidPoints = vtk.vtkPoints()
        cellTypes = vtk.vtkIntArray()
        cellTypes.SetName("CellTypes")
        
#        cellVolumes = vtk.vtkIntArray()
#        cellVolumes.SetName("CellVolumes")
        cellScalars = vtk.vtkFloatArray()
        cellScalars.SetName("CellScalars") # used to store size of the glyph - we are using 0.5 for now
        
        cellTypeIntAddr = self.mdm.extractAddressIntFromVtkObject(cellTypes)
        centroidPointsAddr = self.mdm.extractAddressIntFromVtkObject(centroidPoints)
        cellScalarsAddr = self.mdm.extractAddressIntFromVtkObject(cellScalars)        
        
        
        
        self.mdm.parentWidget.fieldExtractor.fillCellFieldData3D(cellTypeIntAddr ,centroidPointsAddr, cellScalarsAddr)          
        # pt=CA.Point3D()
        # from random import random
        # for x in xrange(fieldDim.x):
            # for y in xrange(fieldDim.y):
                # for z in xrange(fieldDim.z):
                    
                    # pt.x=x
                    # pt.y=y
                    # pt.z=z
                    
                    # cellStack = cellFieldS.get(pt)                  
                    # if not cellStack:
                        # pass
                        # # print 'pt=',pt, ' cellStack=',cellStack
                    # else:
                        # # print 'pt=',pt, ' cellStack=',cellStack
                        # size=cellStack.getFillLevel()
                        
                        # for idx in xrange(size):
                            # cell=cellStack.getCellByIdx(idx)                            
                            # # print 'firstCell.id=',cell.id,' type=',cell.type                    
                            # if size>1:
                                # # centroidPoints.InsertNextPoint(pt.x+random(),pt.y+random(),pt.z+random())
                                # centroidPoints.InsertNextPoint(pt.x+idx/(size*1.0),pt.y+idx/(size*1.0),pt.z+idx/(size*1.0))
                            # else:
                                # centroidPoints.InsertNextPoint(pt.x,pt.y,pt.z)
                                
                            # cellTypes.InsertNextValue(cell.type)                    
                            # cellScalars.InsertNextValue(0.5) 

                        # size=cellStack.getFillLevel()                             
                            
        centroidsPD = vtk.vtkPolyData()
        centroidsPD.SetPoints(centroidPoints)
        centroidsPD.GetPointData().SetScalars(cellTypes)

#        if self.scaleGlyphsByVolume:
        centroidsPD.GetPointData().AddArray(cellScalars)

        centroidGS = vtk.vtkSphereSource()
        thetaRes = Configuration.getSetting("CellGlyphThetaRes")     # todo: make class attrib; update only when changes
        phiRes = Configuration.getSetting("CellGlyphPhiRes")            
        centroidGS.SetThetaResolution(thetaRes)  # increase these values for a higher-res sphere glyph
        centroidGS.SetPhiResolution(phiRes)

        centroidGlyph = vtk.vtkGlyph3D()
        
        if VTK_MAJOR_VERSION>=6:
            centroidGlyph.SetInputData(centroidsPD)
        else:    
            centroidGlyph.SetInput(centroidsPD)        
        
        centroidGlyph.SetSource(centroidGS.GetOutput())
        
        glyphScale = Configuration.getSetting("CellGlyphScale")            
        centroidGlyph.SetScaleFactor( glyphScale )
        centroidGlyph.SetIndexModeToScalar()
        centroidGlyph.SetRange(0,self.mdm.celltypeLUTMax)

        centroidGlyph.SetColorModeToColorByScalar()
#        if self.scaleGlyphsByVolume:
        centroidGlyph.SetScaleModeToScaleByScalar()
        
#        centroidGlyph.SetScaleModeToDataScalingOff()  # call this to disable scaling by scalar value
#        centroidGlyph.SetScaleModeToDataScalingOn()   # method doesn't even exist?!

        centroidGlyph.SetInputArrayToProcess(3,0,0,0,"CellTypes")
        centroidGlyph.SetInputArrayToProcess(0,0,0,0,"CellScalars")

        if VTK_MAJOR_VERSION>=6:
            self.mdm.cellGlyphsMapper.SetInputData(centroidGlyph.GetOutput())
        else:    
            self.mdm.cellGlyphsMapper.SetInput(centroidGlyph.GetOutput())
        
        self.mdm.cellGlyphsMapper.SetScalarRange(0,self.mdm.celltypeLUTMax)
        self.mdm.cellGlyphsMapper.ScalarVisibilityOn()
        
        self.mdm.cellGlyphsMapper.SetLookupTable(self.mdm.celltypeLUT)   # defined in parent class
#        print MODULENAME,' usedCellTypesList=' ,self.usedCellTypesList

        _glyphActor.SetMapper(self.mdm.cellGlyphsMapper)  # Note: we don't need to scale actor for hex lattice here since using cell info
