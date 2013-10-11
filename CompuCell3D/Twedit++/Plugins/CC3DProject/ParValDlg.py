import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_parvaldlg
import sys
import os.path
from CC3DProject.enums import *

MAC = "qt_mac_set_native_menubar" in dir()


class ParValDlg(QDialog,ui_parvaldlg.Ui_ParValDlg):
    #signals
    # gotolineSignal = QtCore.pyqtSignal( ('int',))
    
    def __init__(self,parent=None):
        super(ParValDlg, self).__init__(parent)
        
        self.setupUi(self)
          
        self.updateUi()
        self.generatePB.clicked.connect(self.__generateValues)
        
        self.typeCB.currentIndexChanged.connect(self.__changeValueType)
        self.valueType=str(self.typeCB.currentText())
     
    def __changeValueType(self,_index):
                
        typeStr=str(self.typeCB.itemText(_index))
        
        values=self.getValues(_castToType=self.valueType) # uses previous type - before change
        
        if typeStr=='float':
            values=map(float,values)
        elif typeStr=='int':
            values=map(int,values)
            
        self.valuesLE.setText(','.join(map(str,values)))
        
        self.valueType=str(self.typeCB.currentText()) # after sucessful type change we store new type 
        
    def setAutoMinMax(self,_val):
        minVal=0.2*_val
        maxVal=2.0*_val
        
        if _val<0:
            minVal,maxVal=maxVal,minVal
        self.minLE.setText(str(minVal))
        self.maxLE.setText(str(maxVal))
        
    def getValueType(self):
        '''returns string denoting type of the values in the generated list'''
        return str(self.typeCB.currentText())
    
    def getValues(self,_castToType=''):
        '''returns list of numerical values for parameter scan'''    
        valueStr=str(self.valuesLE.text())
        from ParameterScanUtils import removeWhiteSpaces
        valueStr=removeWhiteSpaces(valueStr)
        
        values=[]
        if valueStr[-1]==',':
            valueStr=valueStr[:-1]
            
        if len(valueStr):            
            values=valueStr.split(',')
            
        typeToCompare=str(self.typeCB.currentText())
        if _castToType:
            typeToCompare=_castToType
        
        if len(values):
            if typeToCompare=='float':
                values=map(float,values)
            elif typeToCompare=='int':   
                values=map(int,values)
                    
        return values    
            
    
    def __generateValues(self):
        print 'INSIDE GENERATE VALUES'
        # try:
        minVal=float(str(self.minLE.text()))
        maxVal=float(str(self.maxLE.text()))
        steps=int(str(self.stepsLE.text()))
        type=str(self.typeCB.currentText())
        distr=str(self.distrCB.currentText())
        # except:
            # return
            
        customValues=[]
        
        if distr=='linear':
            if steps>1:
                interval=(maxVal-minVal)/float(steps-1)
                values=[minVal+i*interval for i in range(steps)]                
            else:
                values=[minVal]
            
        if type=='int':        
            values=map(int,values)
            
        values=map(str,values)        # convert to string list 
        valuesStr=','.join(values)    
        
        self.valuesLE.setText(valuesStr)
        
    def updateUi(self):
        pass
    
        
        

