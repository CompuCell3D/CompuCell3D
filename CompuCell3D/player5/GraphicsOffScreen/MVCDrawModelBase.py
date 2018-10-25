import vtk
import Configuration
import string
import CompuCellSetup

VTK_MAJOR_VERSION=vtk.vtkVersion.GetVTKMajorVersion()
MODULENAME='----- MVCDrawModelBase.py: '
class MVCDrawModelBase:
    def __init__(self):
        
        (self.minCon, self.maxCon) = (0, 0)

        from weakref import ref
        # # # self.graphicsFrameWidget=graphicsFrameWidget
        # # # self.qvtkWidget=self.graphicsFrameWidget.qvtkWidget
        self.currentDrawingFunction=None       
        self.fieldTypes=None 
        self.currentDrawingParameters=None
        self.field_extractor = None

        self.cellType = None
        self.cell_id_array = None
        self.used_cell_types_list = None
        self.lattice_type = None
        self.lattice_type_str = None
#        self.scaleGlyphsByVolume = False
        
        # should also set "periodic" boundary condition flag(s) (e.g. for drawing FPP links that wraparound)

    def init_lattice_type(self):
        """
        Initializes lattice type and lattice type enum
        :return: None
        """
        self.lattice_type_str = CompuCellSetup.ExtractLatticeType()

        if self.lattice_type_str in Configuration.LATTICE_TYPES.keys():
            self.lattice_type = Configuration.LATTICE_TYPES[self.lattice_type_str]
        else:
            # default choice
            self.lattice_type = Configuration.LATTICE_TYPES["Square"]

    def get_lattice_type(self):
        """
        Returns lattice type as str
        :return: {str} lattice type str
        """
        if self.lattice_type is None:
            self.init_lattice_type()
        return self.lattice_type

    def get_lattice_type_str(self):
        """
        Returns lattice type as integer
        :return: {int} enum corresponding to lattice type
        """
        if self.lattice_type_str is None:
            self.init_lattice_type()
        return self.lattice_type_str

    def set_cell_field_data(self, cell_field_data_dict):
        """
        Stores information about cell field as class variables
        :param cell_field_data_dict: {dict}
        :return:
        """

        self.cellType = cell_field_data_dict['cell_type_array']
        self.cell_id_array = cell_field_data_dict['cell_id_array']
        self.used_cell_types_list = cell_field_data_dict['used_cell_types']

    def setDrawingParametersObject(self,_drawingParams):
        self.currentDrawingParameters=_drawingParams
        
    def setDrawingParameters(self,_bsd,_plane,_planePos,_fieldType):   
        self.bsd=_bsd
        self.plane=_plane
        self.planePos=_planePos
        self.fieldtype=_fieldType
        
    def setDrawingFunctionName(self,_fcnName):
        # print "\n\n\n THIS IS _fcnName=",_fcnName," self.drawingFcnName=",self.drawingFcnName
        
        if self.drawingFcnName != _fcnName:
            self.drawingFcnHasChanged=True
        else:
            self.drawingFcnHasChanged=False
        self.drawingFcnName=_fcnName
        
    def clearDisplay(self):
        print MODULENAME,"     clearDisplay() "
        for actor in self.currentActors:
            self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actor])
            
        self.currentActors.clear()
    
    def Render(self):   # never called?!
#        print MODULENAME,"     --------- Render() "
        self.graphicsFrameWidget.Render()

    def get_cell_actors_metadata(self): pass

    def get_min_max_metadata(self, scene_metadata, field_name):
        """
        Returns dictionary with the following entries:
        1. MinRangeFixed
        2. MaxRangeFixed
        3. MinRange
        3. MaxRange

        :param scene_metadata:{dict} metadata dictionary
        :param field_name: {str} field name
        :return: {dict}
        """
        out_dict = {}
        if set(['MinRangeFixed',"MaxRangeFixed",'MinRange','MaxRange']).issubset( set(scene_metadata.keys())):

            min_range_fixed = scene_metadata['MinRangeFixed']
            max_range_fixed = scene_metadata['MaxRangeFixed']
            min_range = scene_metadata['MinRange']
            max_range = scene_metadata['MaxRange']
        else:
            min_range_fixed = Configuration.getSetting("MinRangeFixed", field_name)
            max_range_fixed = Configuration.getSetting("MaxRangeFixed", field_name)
            min_range = Configuration.getSetting("MinRange", field_name)
            max_range = Configuration.getSetting("MaxRange", field_name)

        out_dict['MinRangeFixed'] = min_range_fixed
        out_dict['MaxRangeFixed'] = max_range_fixed
        out_dict['MinRange'] = min_range
        out_dict['MaxRange'] = max_range

        return out_dict

    def init_vector_field_actors(self, actor_specs, drawing_params=None):
        """
        initializes vector field actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """
        raise NotImplementedError()


    def init_cluster_border_actors(self, actor_specs, drawing_params=None):
        """
        initializes cluster border actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """
        raise NotImplementedError()

    def init_fpp_links_actors(self, actor_specs, drawing_params=None):
        """
        initializes fpp links actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """
        raise NotImplementedError()

    def init_concentration_field_actors(self, actor_specs, drawing_params=None):
        """
        initializes concentration field actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """
        raise NotImplementedError()

    def init_legend_actors(self, actor_specs, drawing_params=None):
        """
        initializes legend (for concentration fields) actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """
        try:
            mapper = actor_specs.metadata['mapper']
        except KeyError:
            print 'Could not find mapper object to draw legend'
            return

        actors_dict = actor_specs.actors_dict

        legend_actor = actors_dict['legend_actor']


        legend_actor.SetLookupTable(mapper.GetLookupTable())
        legend_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        legend_actor.GetPositionCoordinate().SetValue(0.01, 0.1)
        legend_actor.SetOrientationToHorizontal()

        legend_actor.SetOrientationToVertical()
        # self.legendActor.SetWidth(0.8)
        # self.legendActor.SetHeight(0.10)

        legend_actor.SetWidth(0.1)
        legend_actor.SetHeight(0.9)

        if VTK_MAJOR_VERSION >= 6:
            legend_actor.SetTitle('')

        # You don't actually need to make contrast for the text as
        # it has shadow!
        text_property = legend_actor.GetLabelTextProperty()
        text_property.SetFontSize(12)  # For some reason it doesn't make effect
        # text.BoldOff()
        text_property.SetColor(1.0, 1.0, 1.0)

        legend_actor.SetLabelTextProperty(text_property)

    def init_cell_field_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell field actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        raise NotImplementedError()

    def init_outline_actors(self, actor_specs, drawing_params=None):
        raise NotImplementedError()

    
    # def initConFieldActors(self, _actors): pass
    #
    # def initVectorFieldCellLevelActors(self, _fillVectorFieldFcn, _actors): pass
    #
    # def initVectorFieldActors(self, _actors): pass
    #
    # def initScalarFieldCellLevelActors(self, _actors): pass
    #
    # def initScalarFieldActors(self, _fillScalarField, _actors): pass
    
    def prepareOutlineActors(self,_actors):pass        
    
    def setContourColor(self):
#        foo=1/0
        color = Configuration.getSetting("ContourColor")
        r = color.red()
        g = color.green()
        b = color.blue()
#        print MODULENAME,'  setBorderColor():   r,g,b=',r,g,b
        self.contourActor.GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))
#        self.contourActorHex.GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))

    def getIsoValues(self,conFieldName):
        self.isovalStr = Configuration.getSetting("ScalarIsoValues",conFieldName)
#        print MODULENAME, '  type(self.isovalStr)=',type(self.isovalStr)
#        print MODULENAME, '  self.isovalStr=',self.isovalStr
        if type(self.isovalStr) == QVariant:
#          isovalStr = isovalStr.toString()
#          print MODULENAME, ' self.isovalStr.toList()=',self.isovalStr.toList()
#          print MODULENAME, ' self.isovalStr.toString()=',self.isovalStr.toString()
          self.isovalStr = str(self.isovalStr.toString())
#          print MODULENAME, ' new type(self.isovalStr)=',type(self.isovalStr)
#        elif type(self.isovalStr) == QString:
        else:
          self.isovalStr = str(self.isovalStr)


#        print MODULENAME, '  pre-replace,split; initScalarFieldDataActors(): self.isovalStr=',self.isovalStr
#        import string
        self.isovalStr = string.replace(self.isovalStr,","," ")
        self.isovalStr = string.split(self.isovalStr)
#        print MODULENAME, '  initScalarFieldDataActors(): final type(self.isovalStr)=',type(self.isovalStr)
#        print MODULENAME, '  initScalarFieldDataActors(): final self.isovalStr=',self.isovalStr

#        print MODULENAME, '  initScalarFieldDataActors(): len(self.isovalStr)=',len(self.isovalStr)
        printIsoValues = False
#        if printIsoValues:  print MODULENAME, ' isovalues= ',
        isoNum = 0
        self.isoValList = []
        for idx in xrange(len(self.isovalStr)):
#            print MODULENAME, '  initScalarFieldDataActors(): idx= ',idx
            try:
                isoVal = float(self.isovalStr[idx])
                if printIsoValues:  print MODULENAME, '  initScalarFieldDataActors(): setting (specific) isoval= ',isoVal
                self.isoValList.append(isoVal)
#                isoContour.SetValue(isoNum, isoVal)
                isoNum += 1
            except:
                print MODULENAME, '  initScalarFieldDataActors(): cannot convert to float: ',self.isovalStr[idx]
                
#        return [1.1,2.2,3.3]
#        print MODULENAME, '  returning self.isoValList=',self.isoValList
        return self.isoValList
    
    # def showContours(self, enable): pass
    
    # def setPlane(self, plane, pos): pass
    
    # def getPlane(self):
        # return ("",0)
    
    def getCamera(self):
        return self.ren.GetActiveCamera()
        
    # def initSimArea(self, _bsd):
        # fieldDim   = _bsd.fieldDim
        # # sim.getPotts().getCellFieldG().getDim()
        # self.setCamera(fieldDim)
       
    def configsChanged(self): pass
        
    # Transforms interval [0, 255] to [0, 1]
    def toVTKColor(self, val):
        return float(val)/255

    def largestDim(self, dim):
        ldim = dim[0]
        for i in range(len(dim)):
            if dim[i] > ldim:
                ldim = dim[i]
                
        return ldim
    
    def setParams(self):
        # You can use either Build() method (256 color by default) or
        # SetNumberOfTableValues() to allocate much more colors!
        self.celltypeLUT = vtk.vtkLookupTable()
        # You need to explicitly call Build() when constructing the LUT by hand     
        self.celltypeLUT.Build()
        self.populateLookupTable()
        # self.dim = [100, 100, 1] # Default values
        
        # for FPP links (and offset also for cell glyphs)
        self.eps = 1.e-4     # not sure how small this should be (checking to see if cell volume -> 0)
        self.stubSize = 3.0  # dangling line stub size for lines that wraparound periodic BCs
#        self.offset = 1.0    # account for fact that COM of cell is offset from visualized lattice
#        self.offset = 0.0    # account for fact that COM of cell is offset from visualized lattice

        # scaling factors to map square lattice to hex lattice (rf. CC3D Manual)
        self.xScaleHex = 1.0
        self.yScaleHex =  0.866
        self.zScaleHex =  0.816
        
        self.lutBlueRed = vtk.vtkLookupTable()
        self.lutBlueRed.SetHueRange(0.667,0.0)
        self.lutBlueRed.Build()
    
    def populateLookupTable(self):
#        print MODULENAME,' populateLookupTable()'
        colorMap = Configuration.getSetting("TypeColorMap")
#        print MODULENAME,' populateLookupTable():  len(colorMap)=',len(colorMap)
        self.celltypeLUT.SetNumberOfTableValues(len(colorMap))
        self.celltypeLUT.SetNumberOfColors(len(colorMap))
#        lutGlyph.SetTableValue(5, 1,0,0, 1.0)     # SetTableValue (vtkIdType indx, double r, double g, double b, double a=1.0)
#        lutGlyph.SetTableValue(8, 0,1,1, 1.0)     # SetTableValue (vtkIdType indx, double r, double g, double b, double a=1.0)
        for key in colorMap.keys():
            r = colorMap[key].red()
            g = colorMap[key].green()
            b = colorMap[key].blue()
            self.celltypeLUT.SetTableValue(key, self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b), 1.0)
#            print "       type=",key," red=",r," green=",g," blue=",b
#            print "       type=",key," (VTK) red=",self.toVTKColor(r)," green=",self.toVTKColor(g)," blue=",self.toVTKColor(b)
        # self.qvtkWidget.repaint()
        self.celltypeLUT.Build()
        self.celltypeLUTMax = self.celltypeLUT.GetNumberOfTableValues() - 1   # cell types = [0,max]
        self.celltypeLUT.SetTableRange(0,self.celltypeLUTMax)
#        print "       celltypeLUTMax=",self.celltypeLUTMax
        
        # self.graphicsFrameWidget.Render()
        
    # Do I need this method?
    # Calculates min and max concentration
    def findMinMax(self, conField, dim):
        import CompuCell
        pt = CompuCell.Point3D() 

        maxCon = 0
        minCon = 0
        for k in range(dim[2]):
            for j in range(dim[1]):
                for i in range(dim[0]):
                    pt.x = i
                    pt.y = j
                    pt.z = k
                    
                    con = float(conField.get(pt))
                    
                    if maxCon < con:
                        maxCon = con
                    
                    if minCon > con:
                        minCon = con

        # Make sure that the concentration is positive
        if minCon < 0:
            minCon = 0

        return (minCon, maxCon)

    # Just returns min and max concentration
    def conMinMax(self):
        return (self.minCon, self.maxCon)
    
    def frac(self, con, minCon, maxCon):
        if maxCon == minCon:
            return 0.0
        else:
            frac = (con - minCon)/(maxCon - minCon)
            
        if frac > 1.0:
            frac = 1.0
            
        if frac < 0.0:
            frac = 0.0

        return frac

    def prepareAxesActors(self, _mappers, _actors):
        pass

    def prepareLegendActors(self, _mappers, _actors):
        legendActor=_actors[0]
        mapper=_mappers[0]
            
        legendActor.SetLookupTable(mapper.GetLookupTable())    
        legendActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        legendActor.GetPositionCoordinate().SetValue(0.01, 0.1)
        legendActor.SetOrientationToHorizontal()
        
        legendActor.SetOrientationToVertical()
        # self.legendActor.SetWidth(0.8)
        # self.legendActor.SetHeight(0.10)

        legendActor.SetWidth(0.1)
        legendActor.SetHeight(0.9)
        
        if VTK_MAJOR_VERSION>=6:
            legendActor.SetTitle('')

        # You don't actually need to make contrast for the text as
        # it has shadow!
        text_property = legendActor.GetLabelTextProperty()
        text_property.SetFontSize(12) # For some reason it doesn't make effect
        # text.BoldOff()
        text_property.SetColor(1.0, 1.0, 1.0)

        legendActor.SetLabelTextProperty(text_property)


#     # Break the settings read into groups?
# #    def readSettings_old(self):   # not ever called?!  (rf. MVCDrawViewBase)
# #        self.readColorsSets()
# #        self.readViewSets()
# #        self.readColormapSets()
# #        self.readOutputSets()
# #        self.readVectorSets()
# #        self.readVisualSets()
#         # simDefaults?
#
#     def readColorsSets(self):
#         #colorsDefaults
#         self._colorMap     = Configuration.getSetting("TypeColorMap")
#         self._borderColor  = Configuration.getSetting("BorderColor")
#         self._contourColor = Configuration.getSetting("ContourColor")
#         self._brushColor   = Configuration.getSetting("BrushColor")
#         self._penColor     = Configuration.getSetting("PenColor")
#
#     def readViewSets(self):
#         # For 3D only?
#         # viewDefaults
#         self._types3D      = Configuration.getSetting("Types3DInvisible")
#
# #    def readColormapSets(self):   # don't think this is ever called
# #        print MODULENAME,' readColormapSets():  doing Config-.getSetting...'
# #        # colormapDefaults
# #        self._minCon       = Configuration.getSetting("minRange")
# #        self._minConFixed  = Configuration.getSetting("minRangeFixed")
# #        self._maxCon       = Configuration.getSetting("MaxRange")
# #        self._maxConFixed  = Configuration.getSetting("MaxRangeFixed")
# #        self._accuracy     = Configuration.getSetting("NumberAccuracy")
# #        self._numLegend    = Configuration.getSetting("NumberOfLegendBoxes")
# #        self._enableLegend = Configuration.getSetting("LegendEnable")
# #        self._contoursOn   = Configuration.getSetting("ContoursOn")
# #        self._numberOfContourLines   = Configuration.getSetting("NumberOfContourLines")
#
#     def readOutputSets(self):
#         # Should I read the settings here?
#         # outputDefaults
#         self._updateScreen     = Configuration.getSetting("ScreenUpdateFrequency")
#         self._imageOutput      = Configuration.getSetting("ImageOutputOn")
#         self._shotFrequency    = Configuration.getSetting("ScreenshotFrequency")
#
#     def readVectorSets(self):
#         # vectorDefaults
#         self._arrowColor   = Configuration.getSetting("ArrowColor")
#         self._arrowLength  = Configuration.getSetting("ArrowLength")
#         self._arrowColorFixed  = Configuration.getSetting("FixedArrowColorOn")
#         self._enableLegendVec  = Configuration.getSetting("LegendEnableVector")
#         self._scaleArrows  = Configuration.getSetting("ScaleArrowsOn")
#         self._accuracyVec  = Configuration.getSetting("NumberAccuracyVector")
#         self._numLegendVec = Configuration.getSetting("NumberOfLegendBoxesVector")
#         self._overlayVec   = Configuration.getSetting("OverlayVectorsOn")
#         self._maxMag       = Configuration.getSetting("MaxMagnitude")
#         self._maxMagFixed  = Configuration.getSetting("MaxMagnitudeFixed")
#         self._minMag       = Configuration.getSetting("MinMagnitude")
#         self._minMagFixed  = Configuration.getSetting("MinMagnitudeFixed")
#
#     def readVisualSets(self):
#         # visualDefaults
#         self._cellBordersOn    = Configuration.getSetting("CellBordersOn")
#         self._clusterBordersOn = Configuration.getSetting("ClusterBordersOn")
#         self._conLimitsOn  = Configuration.getSetting("ConcentrationLimitsOn")
#         self._zoomFactor   = Configuration.getSetting("ZoomFactor")

    def setLatticeType(self, latticeType):
        self.latticeType=latticeType

    
