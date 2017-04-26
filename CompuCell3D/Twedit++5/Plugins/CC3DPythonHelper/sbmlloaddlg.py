import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_sbmlloaddlg
import sys
import string
import os

MAC = "qt_mac_set_native_menubar" in dir()



class SBMLLoadDlg(QDialog,ui_sbmlloaddlg.Ui_SBMLLoadDlg):
    #signals
    # gotolineSignal = QtCore.pyqtSignal( ('int',))
    
    def __init__(self,_currentEditor=None,parent=None):
        super(SBMLLoadDlg, self).__init__(parent)
        self.editorWindow=parent
        self.setupUi(self)
        self.sbmlPath=''
        
        if not MAC:
            self.leaveEmptyPB.setFocusPolicy(Qt.NoFocus)

        self.updateUi()        

    def setCurrentPath(self,_sbmlPath):
        self.sbmlPath=_sbmlPath
    
    @pyqtSignature("") # signature of the signal emited by the button
    def on_browsePB_clicked(self):
        filterList=QString()
        filterList.append("SBML file (*.sbml *.xml);;")            
        filterList.append("All files (*);;")    
        
        fileName = QFileDialog.getOpenFileName(self,"Open SBML file...",self.sbmlPath,filterList)
        fileName=str(fileName)
        if fileName=='':
            return
            
        fileName=os.path.abspath(fileName)
        
        
        os.path.dirname(fileName)
        modelName,extension=os.path.splitext(os.path.basename(fileName))
        modelNickname=modelName[0:3].upper() if len(modelName)>3 else modelName.upper()
        self.modelNameLE.setText(modelName)
        self.modelNicknameLE.setText(modelNickname)
        self.fileNameLE.setText(fileName)
        
            
    # def keyPressEvent(self, event):            

        # molecule=str(self.afMoleculeLE.text())
        # molecule=string.rstrip(molecule)
        # if event.key()==Qt.Key_Return :
            # if molecule!="":
                # self.on_afMoleculeAddPB_clicked()
                # event.accept()

    # @pyqtSignature("") # signature of the signal emited by the button    
    # def on_afMoleculeAddPB_clicked(self):
        
        # molecule=str(self.afMoleculeLE.text())
        # molecule=string.rstrip(molecule)
        # rows=self.afTable.rowCount()
        # if molecule =="":
            # return

        # # check if molecule with this name already exist               
        # moleculeAlreadyExists=False
        # for rowId in range(rows):
            # name=str(self.afTable.item(rowId,0).text())
            # name=string.rstrip(name)
            # if name==molecule:
                # moleculeAlreadyExists=True
                # break
        
        # if moleculeAlreadyExists:
            # QMessageBox.warning(self,"Molecule Name Already Exists","Molecule name already exist. Please choose different name",QMessageBox.Ok)
            # return
            
        # self.afTable.insertRow(rows)        
        # moleculeItem=QTableWidgetItem(molecule)
        # self.afTable.setItem (rows,0,  moleculeItem)
        
        # # reset molecule entry line
        # self.afMoleculeLE.setText("")
        # return 
        
    # @pyqtSignature("") # signature of the signal emited by the button
    # def on_clearAFTablePB_clicked(self):
        # rows=self.afTable.rowCount()
        # for i in range (rows-1,-1,-1):
            # self.afTable.removeRow(i)
                        
    # def extractInformation(self):
        # adhDict={}
        # for row in range(self.afTable.rowCount()):
            # molecule=str(self.afTable.item(row,0).text())
            # adhDict[row]=molecule
            
        # return adhDict,str(self.bindingFormulaLE.text())
        
    def updateUi(self):    
        pass
        
