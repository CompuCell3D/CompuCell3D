
from SBMLSolverHelper import SBMLSolverHelper

class CAPySteppableBase(SBMLSolverHelper):
    def __init__(self,_caManager,_frequency=1):
        SBMLSolverHelper.setModelType('CA')
        SBMLSolverHelper.__init__(self) 
        self.caManager = _caManager
        self.frequency =_frequency
        self.inventory = self.caManager.getCellInventory()        
        self.cellList = CellList(self.inventory)  
        self.cellListByType = CellListByType(self.inventory)          
        self.cellField = self.caManager.getCellFieldS()
        self.dim = self.cellField.getDim()
        
        
    def getDictionaryAttribute(self,_cell):
        # access/modification of a dictionary attached to cell - make sure to decalare in main script that you will use such attribute
        import CA        
        return CA.getPyAttrib(_cell)        
        
    def start(self):
        pass  
                
    def step(self,mcs):
        pass

    def finish(self):
        pass        

    def addNewPlotWindow(self, _title='',_xAxisTitle='',_yAxisTitle='',_xScaleType='linear',_yScaleType='linear'):
        import CompuCellSetup
        return CompuCellSetup.addNewPlotWindow(_title,_xAxisTitle,_yAxisTitle,_xScaleType,_yScaleType)
    


class CellList(object):
    def __init__(self,_inventory):
        self.inventory = _inventory
    def __iter__(self):
        return CellListIterator(self)
    def __len__(self):
        return int(self.inventory.getSize())

class CellListIterator(object):
    def __init__(self, _cellList):
        import CA
        self.inventory = _cellList.inventory
        self.invItr=CA.STLPyIteratorCINV()
        self.invItr.initialize(self.inventory.getContainer())
        self.invItr.setToBegin()
    def next(self):
        if not self.invItr.isEnd():
            self.cell = self.invItr.getCurrentRef()
            self.invItr.next()
            return self.cell
        else:
            raise StopIteration
    def __iter__(self):
            return self
            
            
# iterating ofver inventory of cells of a given type
class CellListByType:
    def __init__(self,_inventory,*args):            
        from CA import mapLongCACellPtr
        from CoreObjects import vectorint 
        self.inventory = _inventory
        
        self.types=vectorint()
        
        self.inventoryByType=mapLongCACellPtr()
        
        self.initTypeVec(args)
        self.inventory.initCellInventoryByMultiType(self.inventoryByType , self.types)  
        
        
        
    def __iter__(self):
        return CellListByTypeIterator(self)

    def __call__(self,*args):
        self.initTypeVec(args)
        self.inventory.initCellInventoryByMultiType(self.inventoryByType , self.types)

        return self       

        
        
    def __len__(self):        
        return int(self.inventoryByType.size())
        
    def initTypeVec(self,_typeList):
        
        self.types.clear()        
        if len(_typeList)<=0:
            self.types.push_back(1) # type 1 
        else:    
            for type in _typeList:
                self.types.push_back(type)
    
    def initializeWithType(self,_type):
        self.types.clear()
        self.types.push_back(_type)
        self.inventory.initCellInventoryByMultiType(self.inventoryByType , self.types)
        
    def refresh(self):
        self.inventory.initCellInventoryByMultiType(self.inventoryByType , self.types)        
        


        
        
class CellListByTypeIterator:
    def __init__(self,  _cellListByType):
        import CA
        self.inventoryByType = _cellListByType.inventoryByType        
        
        self.invItr=CA.mapLongCACellPtrPyItr()
        self.invItr.initialize(self.inventoryByType)        
        self.invItr.setToBegin()
        
    def next(self):
        if not self.invItr.isEnd():
            self.cell=self.invItr.getCurrentRef()
            # print 'self.idCellPair=',self.idCellPair
            # print 'dir(self.idCellPair)=',dir(self.idCellPair)
            self.invItr.next()
            return self.cell
#       
        else:
            raise StopIteration
    
    def __iter__(self):
            return self 
            