import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_parvaldlg
import sys
import os.path
# from CC3DProject.enums import *
from ParameterScanEnums import *

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
        
        self.typeToCBindexDict={FLOAT:0,INT:1,STRING:2} # dict used provide easy mapping between type combo box index and types
        
        from ParameterScanUtils import ParameterScanData
        self.psd=ParameterScanData()
        
    
        
    
    def guessParameterValueType(self,_val):
        ''' this fcn attempts to guess type of the parameter in this order:int,float, default string
        '''
        if _val.find('.')>=0: 
            try:
                float(_val)
                return FLOAT    
            except ValueError,e:
                pass
        
        try:
            int(_val)
            return INT
        except ValueError,e:
            pass
            
            
        return STRING    
            
    def initParameterScanData(self,_parValue,_parName,_parType=XML_CDATA,_parAccessPath=''):
        psd=self.psd
        psd.name=_parName
        psd.type=_parType
        psd.accessPath=_parAccessPath
        
        guessedType=self.guessParameterValueType(_parValue)
        print 'guessedType=',guessedType
        
        self.typeCB.setCurrentIndex(self.typeToCBindexDict[guessedType])
        
        self.setAutoMinMax(_parValue,guessedType)
        
        self.valueType=str(self.typeCB.currentText()) 
        
        # self.recordValues()
        
    def recordValues(self):
        psd=self.psd
        psd.valueType=self.getValueType()
        # VALUE_TYPE_DICT_REVERSE[self.getValueType()]                
        psd.customValues=self.getValues()
    
    def __changeValueType(self,_index):
    
        if not str(self.valuesLE.text()).strip():return 
        
        self.__generateValues()
          
        
    def setAutoMinMax(self,_val,_valType=FLOAT):
        if _valType==STRING:return
        
        val=None
        if _valType==FLOAT:
            val=float(_val)
        elif  _valType==INT:   
            val=float(_val)
            
        
        
        minVal=0.2*val
        maxVal=2.0*val
        
        if val<0:
            minVal,maxVal=maxVal,minVal
            
        if _valType==INT:
            minVal=int(minVal)
            maxVal=int(maxVal)
        
        self.minLE.setText(str(minVal))
        self.maxLE.setText(str(maxVal))
        
    def getValueType(self):
        '''returns string denoting type of the values in the generated list'''
        try:
            return VALUE_TYPE_DICT_REVERSE[str(self.typeCB.currentText())]                
        except LookupError,e:
            return None
        
        # return str(self.typeCB.currentText())
    
    def getValues(self,_castToType=None):
        '''returns list of numerical values for parameter scan'''    
        valueStr=str(self.valuesLE.text())
        from ParameterScanUtils import removeWhiteSpaces
        valueStr=removeWhiteSpaces(valueStr)
        
        values=[]
        
        if valueStr=='':return  values
            
        if valueStr[-1]==',':
            valueStr=valueStr[:-1]
        
        typeToCompare=self.getValueType()
        # str(self.typeCB.currentText())
        if _castToType:
            typeToCompare=_castToType        
        
        # we have to split values differently depending whether they are strings or numbers
        values=None
        
        if  typeToCompare==STRING:    
            from ParameterScanUtils import extractListOfStrings
            values=extractListOfStrings(valueStr)
            
                
        else:
            if len(valueStr):            
                values=valueStr.split(',')
            

        
        if len(values):
            # if typeToCompare=='float':
            if typeToCompare==FLOAT:
                values=map(float,values)
            elif typeToCompare==INT:                   
            # elif typeToCompare=='int':   
                values=map(int,values)
                    
        return values    
            
    
    def __generateValues(self):
        
        try:
            minVal=float(str(self.minLE.text()))
            maxVal=float(str(self.maxLE.text()))
            steps=int(str(self.stepsLE.text()))
            type=self.getValueType()
            # type=str(self.typeCB.currentText())            
            distr=str(self.distrCB.currentText())
        except ValueError,e:
            return
            
        if minVal>maxVal:
            minValStr=str(self.minLE.text())
            maxValStr=str(self.maxLE.text())
            minValStr,maxValStr=maxValStr,minValStr
            
            self.minLE.setText(minValStr)
            self.maxLE.setText(maxValStr)
            
            minVal,maxVal=maxVal,minVal
            
        # except:
            # return
            
        # if type=='string':
        if type==STRING:
            return
            
        values=[]
        
        if distr=='linear':
            if steps>1:
                interval=(maxVal-minVal)/float(steps-1)
                values=[minVal+i*interval for i in range(steps)]                
            else:
                values=[minVal]
                
        elif distr=='random':
            from random import random
            values=[minVal+random()*(maxVal-minVal) for i in range(steps)]                
        elif distr=='log':
            print 'generating log distr'
            if minVal <0. or maxVal<0. :
            
                QMessageBox.warning(self,"Wrong Min/Max values","Please make sure that min and max values are positive for logarithmic distributions")
                return
            from math import log,exp
            minLog,maxLog=log(minVal),log(maxVal)    
            
            if steps>1:
                interval=(maxLog-minLog)/float(steps-1)
                values=[minLog+i*interval for i in range(steps)]                
            else:
                values=[minLog]
                
            values=map(exp,values)
                
            
                
            
        # if type=='int':        
        if type==INT:        
            values=map(int,values)
        

        #remove duplicates fromt he list
        values=list(set(values))
        values.sort()
        
        values=map(str,values)        # convert to string list         
        
        
        
        valuesStr=','.join(values)    
        
        self.valuesLE.setText(valuesStr)        
        self.valueType=str(self.typeCB.currentText()) # after sucessful type change we store new type 
        # self.recordValues()
        
    def updateUi(self):
        pass
    
        
        

