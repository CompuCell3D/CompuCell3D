from PyQt4.QtGui import *
from PyQt4.QtCore import *

(ORGANIZATION, APPLICATION) = ("Biocomplexity", "CompuCellPlayer")

class Settings():
    def __init__(self):
        self.colorMap       = {}
        self.cellsOn        = True
        self.bordersOn      = True
        self.cellGlyphsOn   = False
        self.FPPLinksOn     = False
        self.contoursOn     = False
        self.concentrationLimitsOn    = True
        self.zoomFactor     = None
        self.defaultColor   = None
        self.borderColor    = None
        self.cellGlyphsColor = None
        self.contourColor   = None
        self.arrowColor     = None
        self.types3DInvisibleVec = [] # ?
        self.typePenMap     = {}
        self.typeBrushMap   = {}
        self.typeColorMap   = {}
        self.recentSimulations = []
        self.defaultBrush   = QBrush()
        self.defaultPen     = QPen()
        self.borderPen      = QPen()
        self.cellGlyphsPen  = QPen()
        self.contourPen     = QPen()
        self.arrowPen       = QPen()

        # Graphics unrelated group of settings
        self.curFile        = None
        self.useXMLFileFlag = None
        self.fileXML        = None
        self.screenshotFrequency    = None
        self.screenUpdateFrequency  = None
        self.noOutputFlag   = None
    
        self.minConcentration       = None
        self.minConcentrationFixed  = None
        self.maxConcentration       = None
        self.maxConcentrationFixed  = None
    
        self.inMagnitude            = None
        self.minMagnitudeFixed      = None
        self.maxMagnitude           = None
        self.maxMagnitudeFixed      = None
        self.numberOfLegendBoxes    = None
        self.numberAccuracy         = None
        self.legendEnable           = None
        self.numberOfContourLines   = None        
        self.arrowLength            = None
        self.numberOfLegendBoxesVector  = None       
        self.numberAccuracyVector   = None
        self.legendEnableVector     = None
        self.overlayVectorCellFields = None
        self.scaleArrows            = None
        self.fixedArrowColorFlag    = None
        self.runPythonFlag          = None
        self.closePlayerAfterSimulationDone = None
        
    def readSettings(self):
        settings = QSettings(ORGANIZATION, APPLICATION)
        # Change grouping!
        settings.beginGroup("/DefaultColors") 

        self.cellsOn      = settings.value("/cellsOn",              QVariant(True)).toBool()
        self.bordersOn    = settings.value("/bordersOn",              QVariant(True)).toBool()
        self.cellGlyphsOn = settings.value("/cellGlyphsOn",           QVariant(False)).toBool()
        self.FPPLinksOn   = settings.value("/FPPLinksOn",              QVariant(False)).toBool()

        self.concentrationLimitsOn    = settings.value("/concentrationLimitsOn",  QVariant(False)).toBool()
        self.contoursOn     = settings.value("/contoursOn",             QVariant(False)).toBool()
        self.numberOfContourLines,ok     = settings.value("/numberOfContourLines",             QVariant(0)).toInt()
        self.curFile        = settings.value("/recentFile",             QVariant("cellsort_2D.xml")).toString()
        self.useXMLFileFlag = settings.value("/useXMLFileFlag",         QVariant(False)).toBool()
        self.fileXML        = settings.value("/fileXML",                QVariant("cellsort_2D.xml")).toString()
        
        recentSimulationsList = settings.value("/recentSimulations").toStringList() # QStringList - has to be converted to Python list
        
        self.recentSimulations=[]
        for i in range(recentSimulationsList.count()):
            self.recentSimulations.append(recentSimulationsList[i])

        self.defaultColor   = QColor(settings.value("/pen",             QVariant("white")).toString())
        self.borderColor    = QColor(settings.value("/border",          QVariant("yellow")).toString())
        self.cellGlyphsColor  = QColor(settings.value("/cellGlyph",        QVariant("red")).toString())
        self.contourColor   = QColor(settings.value("/contour",         QVariant("white")).toString())
        self.arrowColor     = QColor(settings.value("/arrowColor",      QVariant("white")).toString())

        self.defaultBrush   .setColor(QColor(settings.value("/brush",   QVariant("white")).toString()))
        self.defaultPen     .setColor(self.defaultColor)
        self.borderPen      .setColor(self.borderColor)
        self.cellGlyphsPen    .setColor(self.cellGlyphsColor)
        self.contourPen     .setColor(self.contourColor)
        self.arrowPen       .setColor(self.arrowColor)
    
        colorList   = QStringList()
        colorList   += "0"
        colorList   += "black"
        colorList   += "1"
        colorList   += "green"
        colorList   += "2"
        colorList   += "blue"
        colorList   += "3"
        colorList   += "red"
        colorList   += "4"
        colorList   += "darkorange"
        colorList   += "5"
        colorList   += "darksalmon"
        colorList   += "6"
        colorList   += "darkviolet"
        colorList   += "7"
        colorList   += "navy"
        colorList   += "8"
        colorList   += "cyan"
        colorList   += "9"
        colorList   += "greenyellow"
        colorList   += "10"
        colorList   += "hotpink"
        
        penColorList = settings.value("/typeColorMap").toStringList()
        if penColorList.count() == 0:
            penColorList = colorList

        k = 0
        for i in range(penColorList.count()/2):
            key, ok = penColorList[k].toInt()
            k       += 1
            value   = penColorList[k]
            k       += 1
            if ok:
                self.typePenMap  [key]  = QPen(QColor(value))
                self.typeBrushMap[key]  = QBrush(QColor(value))
                self.typeColorMap[key]  = QColor(value)
            
    
        self.zoomFactor, ok         = settings.value("/zoomFactor",             QVariant(1)).toInt()
        self.screenshotFrequency, ok    = settings.value("/screenshotFrequency",    QVariant(100)).toInt()
        self.screenUpdateFrequency, ok  = settings.value("/screenUpdateFrequency",  QVariant(10)).toInt()
        self.noOutputFlag           = settings.value("/noOutputFlag",           QVariant(True)).toBool()
    
        self.minConcentration, ok   = settings.value("/minConcentration",       QVariant(0.0)).toDouble()
        self.minConcentrationFixed  = settings.value("/minConcentrationFixed",  QVariant(False)).toBool()
        self.maxConcentration, ok   = settings.value("/maxConcentration",       QVariant(1.0)).toDouble()
        self.maxConcentrationFixed  = settings.value("/maxConcentrationFixed",  QVariant(False)).toBool()
    
        self.minMagnitude, ok       = settings.value("/minMagnitude",           QVariant(0.0)).toDouble()
        self.minMagnitudeFixed      = settings.value("/minMagnitudeFixed",      QVariant(False)).toBool()
        self.maxMagnitude, ok       = settings.value("/maxMagnitude",           QVariant(1.0)).toDouble()
        self.maxMagnitudeFixed      = settings.value("/maxMagnitudeFixed",      QVariant(False)).toBool()
        self.numberOfLegendBoxes, ok = settings.value("/numberOfLegendBoxes",   QVariant(5)).toInt()
        self.numberAccuracy, ok     = settings.value("/numberAccuracy",         QVariant(3)).toInt()
        self.legendEnable           = settings.value("/legendEnable",           QVariant(True)).toBool()
        self.arrowLength, ok        = settings.value("/arrowLength",            QVariant(3)).toInt()
        self.numberOfLegendBoxesVector, ok  = settings.value("/numberOfLegendBoxesVector",  QVariant(5)).toInt()
        self.numberAccuracyVector, ok   = settings.value("/numberAccuracyVector",   QVariant(3)).toInt()
        self.legendEnableVector     = settings.value("/legendEnableVector",     QVariant(True)).toBool()
        self.overlayVectorCellFields    = settings.value("/overlayVectorCellFields",    QVariant(False)).toBool()
        self.scaleArrows            = settings.value("/scaleArrows",            QVariant(False)).toBool()
        self.fixedArrowColorFlag    = settings.value("/fixedArrowColorFlag",    QVariant(False)).toBool()
    
    
        self.runPythonFlag          = settings.value("/runPython",              QVariant(False)).toBool();
        # ? - pyConfData.pythonFileName=settings.value("/pythonFileName","defaultCompuCellScript.py").toString();
        self.closePlayerAfterSimulationDone = settings.value("/closePlayerAfterSimulationDone", QVariant(False)).toBool()
        types3DinvisibleList        = settings.value("/types3DInvisible").toStringList();
        
        for i in range(types3DinvisibleList.count()):
            value, ok   = types3DinvisibleList[i].toInt()
            self.types3DInvisibleVec.append(value)
        
        settings.endGroup()
    
    def writeSettings(self):
        settings = QSettings (ORGANIZATION, APPLICATION)
        
        settings.beginGroup("/DefaultColors")
        settings.setValue("/brush",                 QVariant(self.defaultBrush.color().name()))
        settings.setValue("/pen",                   QVariant(self.defaultPen.color().name()))
        settings.setValue("/border",                QVariant(self.borderPen.color().name()))

        settings.setValue("/cellsOn",               QVariant(self.cellsOn))
        settings.setValue("/bordersOn",             QVariant(self.bordersOn))
        settings.setValue("/cellGlyphsOn",          QVariant(self.cellGlyphsOn))
        settings.setValue("/FPPLinksOn",            QVariant(self.FPPLinksOn))
        settings.setValue("/concentrationLimitsOn", QVariant(self.concentrationLimitsOn))
        
        settings.setValue("/contour",               QVariant(self.contourPen.color().name()))
        settings.setValue("/contoursOn",            QVariant(self.contoursOn ))
        settings.setValue("/numberOfContourLines",  QVariant(self.numberOfContourLines))        
        settings.setValue("/arrowColor",            QVariant(self.arrowPen.color().name()))
    
        # ?
        if self.useXMLFileFlag:
            settings.setValue("/recentFile",        QVariant(self.curFile))
    
        settings.setValue("/useXMLFileFlag",        QVariant(self.useXMLFileFlag))
        settings.setValue("/fileXML",               QVariant(self.fileXML)) # ?
    
    
        recentSimulationsList = QStringList()
        
        for recentSimulation in self.recentSimulations:
            recentSimulationsList.append(recentSimulation)
        settings.setValue("/recentSimulations",          QVariant(recentSimulationsList))
    
    
        penColorList = QStringList()
        
        for i in range(len(self.typePenMap)):
            keys = self.typePenMap.keys()
            penColorList.append(str(keys[i]))
            penColorList.append(str(self.typePenMap[keys[i]].color().name()))
            
        settings.setValue("/typeColorMap",          QVariant(penColorList))
        settings.setValue("/zoomFactor",            QVariant(self.zoomFactor))
        settings.setValue("/screenshotFrequency",   QVariant(int(self.screenshotFrequency)))
        settings.setValue("/screenUpdateFrequency", QVariant(int(self.screenUpdateFrequency)))
        settings.setValue("/noOutputFlag",          QVariant(self.noOutputFlag))
    
        settings.setValue("/minConcentration",      QVariant(self.minConcentration))
        settings.setValue("/minConcentrationFixed", QVariant(self.minConcentrationFixed))
        settings.setValue("/maxConcentration",      QVariant(self.maxConcentration))
        settings.setValue("/maxConcentrationFixed", QVariant(self.maxConcentrationFixed))
    
        settings.setValue("/minMagnitude",          QVariant(self.minMagnitude))
        settings.setValue("/minMagnitudeFixed",     QVariant(self.minMagnitudeFixed))
        settings.setValue("/maxMagnitude",          QVariant(self.maxMagnitude))
        settings.setValue("/maxMagnitudeFixed",     QVariant(self.maxMagnitudeFixed))
        settings.setValue("/numberOfLegendBoxes",   QVariant(int(self.numberOfLegendBoxes)))
        settings.setValue("/numberAccuracy",        QVariant(self.numberAccuracy))
        settings.setValue("/legendEnable",          QVariant(self.legendEnable))
        
        settings.setValue("/arrowLength",           QVariant(self.arrowLength))
        settings.setValue("/numberOfLegendBoxesVector", QVariant(int(self.numberOfLegendBoxesVector)))
        settings.setValue("/numberAccuracyVector",  QVariant(int(self.numberAccuracyVector)))
        settings.setValue("/legendEnableVector",    QVariant(self.legendEnableVector))
        settings.setValue("/overlayVectorCellFields",   QVariant(self.overlayVectorCellFields))
        settings.setValue("/scaleArrows",           QVariant(self.scaleArrows))
        settings.setValue("/fixedArrowColorFlag",   QVariant(self.fixedArrowColorFlag))
    
        settings.setValue("/runPython",             QVariant(self.runPythonFlag))
        # ? settings.setValue("/pythonFileName",        QVariant(self.pyConfData.pythonFileName))
        settings.setValue("/closePlayerAfterSimulationDone",    QVariant(self.closePlayerAfterSimulationDone))
        
        types3DinvisibleList = QStringList()
        for i in range(len(self.types3DInvisibleVec)):
            types3DinvisibleList.append(str(self.types3DInvisibleVec[i]))
            
        settings.setValue("/types3DInvisible",      QVariant(types3DinvisibleList))
        settings.endGroup();
    
    # ?
    def avoidType(type): pass

    def dumpSettings(self):
        s = "colorMap:"
        s += ""
        print s

    def writePlayerSettings(self):
        print MODULENAME, '  writePlayerSettings()'
        print 'screenUpdateFrequency ', self.screenUpdateFrequency
        print 'screenshotFrequency ', self.screenshotFrequency
