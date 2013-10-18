import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_parameterdlg
import sys
import os.path
# from CC3DProject.enums import *
from ParameterScanEnums import *

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
        self.fileType='XML'

        self.scannableParams={}
        self.xmlString=''
        self.accessPath=''
        self.parameterScanXMLElements={}
        self.parameterScanDataMap={}
        
        # self.scannedFileName='' # name of the file being scanned - can be either XML , or Python
        
        self.setupUi(self)
          
        self.updateUi()
        
    def setFileType(self,_type):
        self.fileType=_type
    
    def __handleActionClicked(self):
        senderBtn=self.sender()
        row,col=senderBtn.getPosition()
        print 'clicked row,column=',(row,col)
        from ParameterScanUtils import ParameterScanData
        
        nameItem=self.paramTW.item(row,PARAMETER)
        
        valueItem=self.paramTW.item(row,VALUE)
        
        # value=float(valueItem.text())
        value=str(valueItem.text())
        
        print 'TYPE=',TYPE
        typeItem=self.paramTW.item(row,TYPE)
        type=TYPE_DICT_REVERSE[str(typeItem.text())]
        print 'type=',type,'\n\n\n\n' 
        
        psd=ParameterScanData()
        
        psd.name=str(nameItem.text())
        psd.type=type
        psd.accessPath=self.accessPath
        
        # show param scan values generation dialog
        from ParValDlg import ParValDlg
        parvaldlg=ParValDlg(self)
        parvaldlg.initParameterScanData(_parValue=value,_parName=psd.name,_parType=psd.type,_parAccessPath=psd.accessPath)
        # parvaldlg.setAutoMinMax(value)
        
        if parvaldlg.exec_():
            valueStr=str()
            try:
                psd.customValues=parvaldlg.getValues()
            except ValueError,e:
                QMessageBox.warning(self,"Error Parsing Parameter List","Please make sure that parameter list entries have correct type")
                return
            psd.valueType=parvaldlg.getValueType()
            # VALUE_TYPE_DICT_REVERSE[parvaldlg.getValueType()]
        else:
            #user canceled
            return

        el=psd.toXMLElem()
        
        from ParameterScanUtils import XMLHandler
        xmlHandler=XMLHandler()
        xmlHandler.writeXMLElement(el.CC3DXMLElement) # because ElementCC3D was constructed in Python we need to get C++ object (CC3DXMLElement) from it this is what writeXMLElement expects
        self.parameterScanDataMap[psd.stringHash()]=psd
        
        # print 'xmlElem=',xmlHandler.xmlString
        

    def displayXMLScannableParameters(self,_elem,_accessPath,_parameterScanFile):
        
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
            
            paramTypeItem=QTableWidgetItem(TYPE_DICT[paramProps[1]])
                
            btn = TablePushButton(table)
            
            
            actionItem=None
            btn.setText('Edit...')
            # if paramProps[1]==0:
                # btn.setText('Add To Scan...')
                # # actionItem=QTableWidgetItem('Add To Scan')
            # elif paramProps[1]==1:
                # btn.setText('View/Edit...')
                # # actionItem=QTableWidgetItem('View/Edit...')
                
            table.setItem (currentRow,PARAMETER,  paramNameItem)
            table.setItem (currentRow,VALUE,  paramValueItem)
            table.setItem (currentRow,TYPE,  paramTypeItem)
            
            table.setCellWidget(currentRow, ACTION, btn)
            btn.setPosition(currentRow, ACTION)
            btn.clicked.connect(self.__handleActionClicked)
            
    def displayPythonScannableParameters(self,_pythonLine,_parameterScanFile):        
        print '_pythonLine=',_pythonLine
        from ParameterScanUtils import ParameterScanUtils        
        psu=ParameterScanUtils()
        foundGlobalVar=psu.checkPythonLineForGlobalVariable(_pythonLine)
        print 'foundGlobalVar=',foundGlobalVar
        
    
    def updateUi(self):
        table=self.paramTW
        table.verticalHeader().setDefaultSectionSize(20)
        table.horizontalHeader().setDefaultSectionSize(20)
        
        table.horizontalHeader().setResizeMode(QHeaderView.Stretch)        
    
        
        

