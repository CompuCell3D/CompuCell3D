import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_celltypedlg
import sys
import string

MAC = "qt_mac_set_native_menubar" in dir()



class CellTypeDlg(QDialog,ui_celltypedlg.Ui_CellTypeDlg):
    #signals
    # gotolineSignal = QtCore.pyqtSignal( ('int',))
    
    def __init__(self,_currentEditor=None,parent=None):
        super(CellTypeDlg, self).__init__(parent)
        self.editorWindow=parent
        self.setupUi(self)
                
        if not MAC:
            self.cancelPB.setFocusPolicy(Qt.NoFocus)
        

        self.updateUi()        
        
    def keyPressEvent(self, event):            
        
        cellType=str(self.cellTypeLE.text())
        cellType=string.rstrip(cellType)
    
        if event.key()==Qt.Key_Return :
            if cellType!="":
                self.on_cellTypeAddPB_clicked()
                event.accept()
        
    @pyqtSignature("") # signature of the signal emited by the button
    def on_cellTypeAddPB_clicked(self):
        
        cellType=str(self.cellTypeLE.text())
        cellType=string.rstrip(cellType)
        rows=self.cellTypeTable.rowCount()
        if cellType =="":
            return
            
        # check if cell type with this name already exist               
        cellTypeAlreadyExists=False
        for rowId in range(rows):
            
            name=str(self.cellTypeTable.item(rowId,0).text())
            name=string.rstrip(name)
            print "CHECKING name=",name+"1"," type=",cellType+"1"
            print "name==cellType ",name==cellType
            if name==cellType:
                cellTypeAlreadyExists=True
                break
        print "cellTypeAlreadyExists=",cellTypeAlreadyExists
        if cellTypeAlreadyExists:
            print "WARNING"
            QMessageBox.warning(self,"Cell type name already exists","Cell type name already exist. Please choose different name",QMessageBox.Ok)
            return
            
        self.cellTypeTable.insertRow(rows)        
        cellTypeItem=QTableWidgetItem(cellType)
        self.cellTypeTable.setItem (rows,0,  cellTypeItem)
        
        cellTypeFreezeItem=QTableWidgetItem()
        cellTypeFreezeItem.data(Qt.CheckStateRole)
        if self.freezeCHB.isChecked():
            
            cellTypeFreezeItem.setCheckState(Qt.Checked)
        else:
            cellTypeFreezeItem.setCheckState(Qt.Unchecked)
            
        self.cellTypeTable.setItem (rows,1,  cellTypeFreezeItem)
        # reset cell type entry line
        self.cellTypeLE.setText("")
        return 

    @pyqtSignature("") # signature of the signal emited by the button
    def on_clearCellTypeTablePB_clicked(self):
    
        rows=self.cellTypeTable.rowCount()
        for i in range (rows-1,-1,-1):
            self.cellTypeTable.removeRow(i)
        
        
        #insert Medium    
        self.cellTypeTable.insertRow(0)        
        mediumItem=QTableWidgetItem("Medium")
        self.cellTypeTable.setItem (0,0,  mediumItem)
        mediumFreezeItem=QTableWidgetItem()        
        mediumFreezeItem.data(Qt.CheckStateRole)
        mediumFreezeItem.setCheckState(Qt.Unchecked)
        self.cellTypeTable.setItem (0,1,  mediumFreezeItem)
        
    def extractInformation(self):
        cellTypeDict={}
        for row in range(self.cellTypeTable.rowCount()):
            type=str(self.cellTypeTable.item(row,0).text())
            freeze=False
            if self.cellTypeTable.item(row,1).checkState()==Qt.Checked:
                print "self.cellTypeTable.item(row,1).checkState()=",self.cellTypeTable.item(row,1).checkState()
                freeze=True
            cellTypeDict[row]=[type,freeze]
            
        return cellTypeDict
        
    def updateUi(self):    
        self.cellTypeTable.insertRow(0)        
        mediumItem=QTableWidgetItem("Medium")
        self.cellTypeTable.setItem (0,0,  mediumItem)
        mediumFreezeItem=QTableWidgetItem()        
        mediumFreezeItem.data(Qt.CheckStateRole)
        mediumFreezeItem.setCheckState(Qt.Unchecked)
        self.cellTypeTable.setItem (0,1,  mediumFreezeItem)
    
        baseSize=self.cellTypeTable.baseSize()
        self.cellTypeTable.setColumnWidth (0,baseSize.width()/2)
        self.cellTypeTable.setColumnWidth (1,baseSize.width()/2)
        self.cellTypeTable.horizontalHeader().setStretchLastSection(True)
