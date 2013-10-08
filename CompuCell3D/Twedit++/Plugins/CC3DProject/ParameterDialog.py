import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_parameterdlg
import sys
import os.path

MAC = "qt_mac_set_native_menubar" in dir()

class TablePushButton(QPushButton):
    def __init__(self,_parent=None):
        super(TablePushButton,self).__init__(_parent)
        self.row=-1
        self.col=-1
        
    def setPosition(self,_row,_col):
        self.row=_row
        self.col=_col
        
    def getPosition(self):
        return self.row,self.col

class ParameterDialog(QDialog,ui_parameterdlg.Ui_ParameterDlg):
    #signals
    # gotolineSignal = QtCore.pyqtSignal( ('int',))
    
    def __init__(self,parent=None):
        super(ParameterDialog, self).__init__(parent)

        self.scannableParams={}
        self.xmlString=''
        self.accessPath=''

        self.setupUi(self)
          
        self.updateUi()      
        
    def __handleActionClicked(self):
        senderBtn=self.sender()
        row,col=senderBtn.getPosition()
        print 'clicked row,column=',(row,col)
        from ParameterScanUtils import ParameterScanData
        
        nameItem=self.paramTW.item(row,0)
        valueItem=self.paramTW.item(row,1)
        value=float(valueItem.text())
        
        psd=ParameterScanData()
        
        psd.name=nameItem.text()
        psd.type='float'
        psd.minValue=value
        psd.maxValue=value
        psd.step=1
        psd.accessPath=self.accessPath
        
        el=psd.toXMLElem()
        
        from ParameterScanUtils import XMLHandler
        xmlHandler=XMLHandler()
        xmlHandler.writeXMLElement(el.CC3DXMLElement) # because ElementCC3D was constructed in Python we need to get C++ object (CC3DXMLElement) from it this is what writeXMLElement expects
        
        print 'xmlElem=',xmlHandler.xmlString
        

        
    def  displayRunnableParameters(self,_elem,_accessPath,_parameterScanFile):
        
        self.accessPath=_accessPath
        
        from ParameterScanUtils import ParameterScanUtils
        
        psu=ParameterScanUtils()
        print  'xmlElem=',_elem
        
        self.xmlString='<'+_elem.name+' '
        for key in _elem.attributes.keys():
            self.xmlString+=' '+key+'="'+_elem.attributes[key]+'"'            

        self.xmlString+='>'            
        self.elemLE.setText(self.xmlString)
        
        self.scannableParams=psu.extractXMLScannableParameters(_elem,_parameterScanFile)
        
        table=self.paramTW
        
        for paramName, paramProps in self.scannableParams.iteritems():
            currentRow=table.rowCount() 
            # if table.rowCount()>0 else 0 
            print 'currentRow=',currentRow,'paramName=',paramName
            table.insertRow(currentRow)            
            paramNameItem=QTableWidgetItem(paramName)
            paramValueItem=QTableWidgetItem(paramProps[0])
            
            btn = TablePushButton(table)
            
            
            actionItem=None
            if paramProps[1]==0:
                btn.setText('Add To Scan...')
                # actionItem=QTableWidgetItem('Add To Scan')
            elif paramProps[1]==1:
                btn.setText('View/Edit...')
                # actionItem=QTableWidgetItem('View/Edit...')
                
            table.setItem (currentRow,0,  paramNameItem)
            table.setItem (currentRow,1,  paramValueItem)
            table.setCellWidget(currentRow, 2, btn)
            btn.setPosition(currentRow, 2)
            btn.clicked.connect(self.__handleActionClicked)
            
            

        
        
        
        
        
    def updateUi(self):
        table=self.paramTW
        table.verticalHeader().setDefaultSectionSize(20)
        table.horizontalHeader().setDefaultSectionSize(20)
        
        table.horizontalHeader().setResizeMode(QHeaderView.Stretch)        
    
        
        

