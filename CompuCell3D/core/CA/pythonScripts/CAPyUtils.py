class CASteppableBase(object):
    def __init__(self,_caManager,_frequency=1):
        self.frequency=_frequency
        self.caManager=_caManager
        
        self.inventory=self.caManager.getCellInventory()        
        self.cellList=CellList(self.inventory)
        
    def start(self):
        pass
        
    def step(self,mcs):
        pass

    def finish(self):
        pass
        

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
            
class DemoSteppable(CASteppableBase):
    def __init__(self,_caManager,_frequency=1):
        CASteppableBase.__init__(self,_caManager,_frequency)
        
    def step(self,mcs):
        print 'inside DemoSteppable step fcn , mcs=',mcs 
        print 'len(self.cellList)=',len(self.cellList)    
        for cell in self.cellList:
            print 'cell.id=',cell.id
            