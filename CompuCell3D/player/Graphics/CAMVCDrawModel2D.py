import vtk 

VTK_MAJOR_VERSION=vtk.vtkVersion.GetVTKMajorVersion()

class CAMVCDrawModel2D(object):
    def __init__(self,_masterDrawModel=None):
        self.initMasterDrawModel(_masterDrawModel)
        self.mdm=None
        
    def initMasterDrawModel(self,_masterDrawModel=None):
        if _masterDrawModel is None:
            return
            
        from weakref import ref
        
        MDM = ref(_masterDrawModel)
        self.mdm=MDM()
                
    def initCellGlyphsActor2D(self,cellGlyphActor):
        print '  initCellGlyphsActor2DCA'
        

        
        fieldDim = self.mdm.currentDrawingParameters.bsd.fieldDim
        dimOrder = self.mdm.dimOrder(self.mdm.currentDrawingParameters.plane)
        self.dim = self.mdm.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]         
        
      
        
        plane = self.mdm.currentDrawingParameters.plane
        planePos = self.mdm.currentDrawingParameters.planePos

        
       
        
        print '_plane=',plane
        print '_planePos=',planePos
        caManager = self.mdm.parentWidget.mysim()
        cellField=caManager.getCellField()
        fieldDim=cellField.getDim()
        
        fieldDimVec=[fieldDim.x,fieldDim.y,fieldDim.z]
        
        pointOrderVec=self.mdm.pointOrder(plane)
        dimOrderVec=self.mdm.dimOrder(plane)
        dim=[0,0,0]
        
        dim[0]=fieldDimVec[dimOrderVec[0]];
        dim[1]=fieldDimVec[dimOrderVec[1]];
        dim[2]=fieldDimVec[dimOrderVec[2]];        
        
        
        print 'pointOrderVec=',pointOrderVec
        print 'dimOrderVec=',dimOrderVec
        
        print 'dim=',dim
        

        
        
     
        
        
        import CA
        
        # pt=CA.Point3D()
        ptVec=[0,0,0]        
        

        
        
        cellFieldS=caManager.getCellFieldS()

        centroidPoints = vtk.vtkPoints()
        cellTypes = vtk.vtkIntArray()
        cellTypes.SetName("CellTypes")
        
        cellScalars = vtk.vtkFloatArray()
        cellScalars.SetName("CellScalars") # used for store radius of the glyph
        
        
        
        cellTypeIntAddr = self.mdm.extractAddressIntFromVtkObject(cellTypes)
        centroidPointsAddr = self.mdm.extractAddressIntFromVtkObject(centroidPoints)
        cellScalarsAddr = self.mdm.extractAddressIntFromVtkObject(cellScalars)
        
        
        self.mdm.parentWidget.fieldExtractor.fillCellFieldData2D(cellTypeIntAddr ,centroidPointsAddr, cellScalarsAddr, plane ,   planePos)          
#        cellVolumes = vtk.vtkIntArray()
#        cellVolumes.SetName("CellVolumes")
        # # # from random import random
        # # # for j in xrange(dim[1]+1):
            # # # for i in xrange(dim[0]+1):
                # # # ptVec[0]=i;
                # # # ptVec[1]=j;
                # # # ptVec[2]=planePos;        


                # # # pt.x=ptVec[pointOrderVec[0]];
                # # # pt.y=ptVec[pointOrderVec[1]];
                # # # pt.z=ptVec[pointOrderVec[2]];            
                
                # # # cellStack=cellFieldS.get(pt)                  
                # # # if not cellStack:
                    # # # pass
                    # # # # print 'pt=',pt, ' cellStack=',cellStack
                # # # else:
                    # # # # print 'pt=',pt, ' cellStack=',cellStack
                    # # # # # # cell=cellStack.getCellByIdx(0)                            
                    # # # # # # centroidPoints.InsertNextPoint(i+random(),j+random(),0.0)
                    
                    # # # # # # cellTypes.InsertNextValue(cell.type)                    
                    # # # # # # cellScalars.InsertNextValue(1.0)                     

                    
                    # # # size=cellStack.getFillLevel()
                    
                    # # # for idx in xrange(size):
                        # # # cell=cellStack.getCellByIdx(idx)                            
                        # # # # print 'firstCell.id=',cell.id,' type=',cell.type                    
                        # # # if size>1:
                            # # # centroidPoints.InsertNextPoint(i+random(),j+random(),0.0)
                        # # # else:
                            # # # centroidPoints.InsertNextPoint(i,j,0.0)
                        # # # cellTypes.InsertNextValue(cell.type)                    
                        # # # cellScalars.InsertNextValue(0.5) 
                    
                    # # # # type=cell.type
            
        centroidsPD = vtk.vtkPolyData()
        centroidsPD.SetPoints(centroidPoints)
        centroidsPD.GetPointData().SetScalars(cellTypes)        
        centroidsPD.GetPointData().AddArray(cellScalars)  # scale by ~radius

        centroidGS = vtk.vtkGlyphSource2D()
        centroidGS.SetGlyphTypeToCircle()
        # #centroidGS.SetScale(1)
        # #gs.FilledOff()
        # #gs.CrossOff()

        centroidGlyph = vtk.vtkGlyph3D()
        if VTK_MAJOR_VERSION>=6:
            centroidGlyph.SetInputData(centroidsPD)
        else:    
            centroidGlyph.SetInput(centroidsPD)
        
        centroidGlyph.SetSource(centroidGS.GetOutput())
#        centroidGlyph.SetScaleFactor( 0.2 )  # rwh: should this lattice size dependent or cell vol or ?
        # # # glyphScale = Configuration.getSetting("CellGlyphScale")
        # # # centroidGlyph.SetScaleFactor( glyphScale )
        centroidGlyph.SetScaleFactor( True )
        #centroidGlyph.SetIndexModeToScalar()
        #centroidGlyph.SetRange(0,2)

        #centroidGlyph.SetScaleModeToDataScalingOff()
        centroidGlyph.SetColorModeToColorByScalar()
        centroidGlyph.SetScaleModeToScaleByScalar()
        centroidGlyph.SetRange(0,self.mdm.celltypeLUTMax)

        centroidGlyph.SetInputArrayToProcess(3,0,0,0,"CellTypes")
#        centroidGlyph.SetInputArrayToProcess(0,0,0,0,"CellVolumes")
        centroidGlyph.SetInputArrayToProcess(0,0,0,0,"CellScalars")

        if VTK_MAJOR_VERSION>=6:
            self.mdm.cellGlyphsMapper.SetInputData(centroidGlyph.GetOutput())
        else:    
            self.mdm.cellGlyphsMapper.SetInput(centroidGlyph.GetOutput())
        
        self.mdm.cellGlyphsMapper.SetScalarRange(0,self.mdm.celltypeLUTMax)
        self.mdm.cellGlyphsMapper.ScalarVisibilityOn()
        self.mdm.cellGlyphsMapper.SetLookupTable(self.mdm.celltypeLUT)
        
        cellGlyphActor.SetMapper(self.mdm.cellGlyphsMapper)
            
        
        return


    def initScalarFieldCartesianActors(self,_fieldType,_actorList):
        print '_fieldType=',_fieldType
        conActor = _actorList[0]
        fieldDim = self.mdm.currentDrawingParameters.bsd.fieldDim
        dimOrder = self.mdm.dimOrder(self.mdm.currentDrawingParameters.plane)
        self.dim = self.mdm.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]         
        
      
        
        plane = self.mdm.currentDrawingParameters.plane
        planePos = self.mdm.currentDrawingParameters.planePos
        
        caManager = self.mdm.parentWidget.mysim()
        cellField=caManager.getCellField()
        fieldDim=cellField.getDim()
        
        fieldDimVec=[fieldDim.x,fieldDim.y,fieldDim.z]
        
        pointOrderVec=self.mdm.pointOrder(plane)
        dimOrderVec=self.mdm.dimOrder(plane)
        dim=[0,0,0]
        
        dim[0]=fieldDimVec[dimOrderVec[0]];
        dim[1]=fieldDimVec[dimOrderVec[1]];
        dim[2]=fieldDimVec[dimOrderVec[2]];        
        
        cartesianVertices = [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]
        
        
        print 'pointOrderVec=',pointOrderVec
        print 'dimOrderVec=',dimOrderVec
        
        print 'dim=',dim        
        
        print '_plane=',plane
        print '_planePos=',planePos        
        
        fieldName = _fieldType[0]
        
        conArray = vtk.vtkDoubleArray()
        cartesianCellsArray = vtk.vtkCellArray()
        pointsArray = vtk.vtkPoints()    
	 
        conArrayAddr = self.mdm.extractAddressIntFromVtkObject(conArray)
        cartesianCellsArrayAddr = self.mdm.extractAddressIntFromVtkObject(cartesianCellsArray)
        pointsArrayAddr = self.mdm.extractAddressIntFromVtkObject(pointsArray)     
     
     
        self.mdm.parentWidget.fieldExtractor.fillScalarFieldData2DCartesian(conArrayAddr,cartesianCellsArrayAddr ,pointsArrayAddr , fieldName , plane ,planePos)
        
        # # # from CoreObjects import Point3D
        # # # pt = Point3D()
        # # # ptVec=[0,0,0]             
        
        # # # field = caManager.getConcentrationField(fieldName)
        # # # print 'field = ',field
        
        # # # pc=0        
        # # # for j in xrange(dim[1]):
            # # # for i in xrange(dim[0]):
                # # # ptVec[0]=i
                # # # ptVec[1]=j
                # # # ptVec[2]=planePos       
        
                # # # pt.x=ptVec[pointOrderVec[0]]
                # # # pt.y=ptVec[pointOrderVec[1]]
                # # # pt.z=ptVec[pointOrderVec[2]]            
                
                # # # conc=field.get(pt)    
                # # # for vertex in cartesianVertices:
                    # # # pointsArray.InsertNextPoint(ptVec[0]+vertex[0],ptVec[1]+vertex[1],0.0)
                    
                # # # pc += 4
                
                # # # cartesianCellsArray.InsertNextCell(4)
                # # # cartesianCellsArray.InsertCellPoint(pc-4)
                # # # cartesianCellsArray.InsertCellPoint(pc-3)
                # # # cartesianCellsArray.InsertCellPoint(pc-2)
                # # # cartesianCellsArray.InsertCellPoint(pc-1)                
                
                # # # conArray.InsertNextValue(conc)
                
                # # # # if conc != 0.0:
                    # # # # print 'pt=',pt,' conc=',conc


                    
        range=conArray.GetRange()
        minCon=range[0]
        maxCon=range[1]                    
        cartesianCellsConPolyData=vtk.vtkPolyData()                    
        cartesianCellsConPolyData.GetCellData().SetScalars(conArray)
        cartesianCellsConPolyData.SetPoints(pointsArray)
        cartesianCellsConPolyData.SetPolys(cartesianCellsArray)
        
        
        
        if VTK_MAJOR_VERSION>=6:
            self.mdm.conMapper.SetInputData(cartesianCellsConPolyData)
        else:    
            self.mdm.conMapper.SetInput(cartesianCellsConPolyData)
        
        self.mdm.conMapper.ScalarVisibilityOn()
        self.mdm.conMapper.SetLookupTable(self.mdm.clut)
        self.mdm.conMapper.SetScalarRange(minCon, maxCon)
        
        conActor.SetMapper(self.mdm.conMapper)                        
                    
                