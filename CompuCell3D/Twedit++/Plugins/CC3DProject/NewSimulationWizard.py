import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_newsimulationwizard
import string
import sys
import os.path

MAC = "qt_mac_set_native_menubar" in dir()

class NewSimulationWizard(QWizard,ui_newsimulationwizard.Ui_NewSimulationWizard):
    #signals
    # gotolineSignal = QtCore.pyqtSignal( ('int',))
    
    def __init__(self,parent=None):
        super(NewSimulationWizard, self).__init__(parent)
        self.cc3dProjectTreeWidget=parent
        self.plugin = self.cc3dProjectTreeWidget.plugin
        # there are issues with Drawer dialog not getting focus when being displayed on linux
        # they are also not positioned properly so, we use "regular" windows 
        if sys.platform.startswith('win'): 
            self.setWindowFlags(Qt.Drawer) # dialogs without context help - only close button exists
        # self.gotolineSignal.connect(self.editorWindow.goToLine)
        
        self.mainProjDir=""
        self.simulationFilesDir=""
        
        
        self.projectPath=""
        self.setupUi(self)
        #This dictionary holds references to certain pages e.g. plugin configuration pages are inserten on demand and access to those pages is facilitated via self.pageDict
        self.pageDict={}
        self.updateUi()

        self.typeTable=[]
        self.diffusantDict={}  
        self.chemotaxisData={}
        
        #we are not really using plot table for Python visualization fields 
        self.plotTable.setHidden(True)
        self.clearPlotTablePB.setHidden(True)
        self.label_14.setHidden(True)
        self.plotLE.setHidden(True)
        self.label_15.setHidden(True)
        self.plotTypeCB.setHidden(True)
        self.plotAddPB.setHidden(True)
        
        if sys.platform.startswith('win'):
            self.setWizardStyle(QWizard.ClassicStyle)
    
    def keyPressEvent(self, event):            
        
        if self.currentPage()==self.pageDict["CellType"][0]:
            cellType=str(self.cellTypeLE.text())
            cellType=string.rstrip(cellType)
        
            if event.key()==Qt.Key_Return :
                if cellType!="":
                    self.on_cellTypeAddPB_clicked()
                    event.accept()
                else:
                    nextButton=self.button(QWizard.NextButton)
                    nextButton.emit(SIGNAL("clicked(bool)") , True)
        elif self.currentPage()==self.pageDict["Diffusants"][0]:
            fieldName=str(self.fieldNameLE.text())
            fieldName=string.rstrip(fieldName)
        
            if event.key()==Qt.Key_Return :
                if fieldName!="":
                    self.on_fieldAddPB_clicked()
                    event.accept()
                else:
                    nextButton=self.button(QWizard.NextButton)
                    nextButton.emit(SIGNAL("clicked(bool)") , True)
                    
        elif self.currentPage()==self.pageDict["ContactMultiCad"][0]:
            cadherin=str(self.cmcMoleculeLE.text())
            cadherin=string.rstrip(cadherin)
            if event.key()==Qt.Key_Return :
                if cadherin!="":
                    self.on_cmcMoleculeAddPB_clicked()
                    event.accept()
                else:
                    nextButton=self.button(QWizard.NextButton)
                    nextButton.emit(SIGNAL("clicked(bool)") , True)
        elif self.currentPage()==self.pageDict["AdhesionFlex"][0]:
            molecule=str(self.afMoleculeLE.text())
            molecule=string.rstrip(molecule)
            if event.key()==Qt.Key_Return :
                if molecule!="":
                    self.on_afMoleculeAddPB_clicked()
                    event.accept()
                else:
                    nextButton=self.button(QWizard.NextButton)
                    nextButton.emit(SIGNAL("clicked(bool)") , True)
            
        elif self.currentPage()==self.pageDict["FinalPage"][0]: # last page
            if event.key()==Qt.Key_Return:
                finishButton=self.button(QWizard.FinishButton)
                finishButton.emit(SIGNAL("clicked(bool)") , True)            
        else:
            if event.key()==Qt.Key_Return:
                # move to the next page
                nextButton=self.button(QWizard.NextButton)
                print "nextButton=",nextButton
                nextButton.emit(SIGNAL("clicked(bool)") , True)
            # nextButton.emit(clicked,True)
            
            pass
            
        # event.ignore()

    # @pyqtSignature("") # signature of the signal emited by the button
    # def on_okButton_clicked(self):
        # self.findChangedConfigs()        
        # self.close()
    @pyqtSignature("") # signature of the signal emited by the button
    def on_piffPB_clicked(self):
        fileName=QFileDialog.getOpenFileName ( self, "Choose PIFF file...")
        fileName=str(fileName)        
        fileName=os.path.abspath(fileName) # normalizing path
        self.piffLE.setText(fileName)
        
    def hideConstraintFlexOption(self):
        self.volumeFlexCHB.setChecked(False)
        self.volumeFlexCHB.setHidden(True)
        self.surfaceFlexCHB.setChecked(False)
        self.surfaceFlexCHB.setHidden(True)
        
    def showConstraintFlexOption(self):        
        if not self.growthCHB.isChecked() and not self.mitosisCHB.isChecked() and not self.deathCHB.isChecked():
            self.volumeFlexCHB.setHidden(False)        
            self.surfaceFlexCHB.setHidden(False)

    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_extPotCHB_toggled(self,_flag):
        if _flag:
            self.extPotLocalFlexCHB.setChecked(not _flag)

    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_extPotLocalFlexCHB_toggled(self,_flag):
        if _flag:
            self.extPotCHB.setChecked(not _flag)

    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_volumeFlexCHB_toggled(self,_flag):
        if _flag:
            self.volumeLocalFlexCHB.setChecked(not _flag)

    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_volumeLocalFlexCHB_toggled(self,_flag):
        if _flag:
            self.volumeFlexCHB.setChecked(not _flag)

    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_surfaceFlexCHB_toggled(self,_flag):
        if _flag:
            self.surfaceLocalFlexCHB.setChecked(not _flag)

    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_surfaceLocalFlexCHB_toggled(self,_flag):
        if _flag:
            self.surfaceFlexCHB.setChecked(not _flag)
            
    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_connectGlobalCHB_toggled(self,_flag):
        if _flag:
            self.connect2DCHB.setChecked(not _flag)
            self.connectGlobalByIdCHB.setChecked(not _flag)

    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_connect2DCHB_toggled(self,_flag):
        if _flag:
            self.connectGlobalCHB.setChecked(not _flag)
            self.connectGlobalByIdCHB.setChecked(not _flag)

    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_connectGlobalByIdCHB_toggled(self,_flag):
        if _flag:
            self.connect2DCHB.setChecked(not _flag)
            self.connectGlobalCHB.setChecked(not _flag)

            
    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_lengthConstraintCHB_toggled(self,_flag):
        if _flag:
            self.lengthConstraintLocalFlexCHB.setChecked(not _flag)

    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_lengthConstraintLocalFlexCHB_toggled(self,_flag):
        if _flag:
            self.lengthConstraintCHB.setChecked(not _flag)
            
            
            
    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_growthCHB_toggled(self,_flag):
        if _flag:
            self.hideConstraintFlexOption()
        else:
            self.showConstraintFlexOption()
            
    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_mitosisCHB_toggled(self,_flag):
        if _flag:
            self.hideConstraintFlexOption()
        else:
            self.showConstraintFlexOption()
            
    @pyqtSignature("bool") # signature of the signal emited by the button
    def on_deathCHB_toggled(self,_flag):
        if _flag:
            self.hideConstraintFlexOption()
        else:
            self.showConstraintFlexOption()

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
        
    @pyqtSignature("") # signature of the signal emited by the button
    def on_fieldAddPB_clicked(self):
        
        fieldName=str(self.fieldNameLE.text())
        fieldName=string.rstrip(fieldName)
        rows=self.fieldTable.rowCount()
        if fieldName =="":
            return
            
        # check if cell type with this name already exist               
        fieldAlreadyExists=False
        for rowId in range(rows):
            
            name=str(self.fieldTable.item(rowId,0).text())
            name=string.rstrip(name)
            print "CHECKING name=",name+"1"," type=",fieldName+"1"
            print "name==cellType ",name==fieldName
            if name==fieldName:
                fieldAlreadyExists=True
                break
        print "fieldAlreadyExists=",fieldAlreadyExists
        if fieldAlreadyExists:
            print "WARNING"
            QMessageBox.warning(self,"Field name name already exists","Field name name already exist. Please choose different name",QMessageBox.Ok)
            return
            
        self.fieldTable.insertRow(rows)        
        fieldNameItem=QTableWidgetItem(fieldName)
        self.fieldTable.setItem (rows,0,  fieldNameItem)
        #picking solver name
        solverName=string.rstrip(str(self.solverCB.currentText()))
        solverNameItem=QTableWidgetItem(solverName)
        self.fieldTable.setItem (rows,1,  solverNameItem)
        
        
        # reset cell type entry line
        
        
        self.fieldNameLE.setText("")
        print "CLEAN FIELD NAME"
        return 
        
    @pyqtSignature("") # signature of the signal emited by the button
    def on_clearFieldTablePB_clicked(self):
    
        rows=self.fieldTable.rowCount()
        for i in range (rows-1,-1,-1):
            self.fieldTable.removeRow(i)
            
    #SECRETION            
    @pyqtSignature("bool") # signature of the signal emited by the button    
    def on_secrConstConcRB_toggled(self,_flag):
        if _flag:           
            self.secrRateLB.setText("Const. Concentration")
        else:
            self.secrRateLB.setText("Secretion Rate")

    @pyqtSignature("bool") # signature of the signal emited by the button    
    def on_secrOnContactRB_toggled(self,_flag):
        if _flag:           
            self.secrAddOnContactPB.setHidden(False)
            self.secrOnContactCellTypeCB.setHidden(False)
            self.secrOnContactLE.setHidden(False)        
        else:
            self.secrAddOnContactPB.setHidden(True)
            self.secrOnContactCellTypeCB.setHidden(True)
            self.secrOnContactLE.setHidden(True)

            
    @pyqtSignature("") # signature of the signal emited by the button    
    def on_secrAddOnContactPB_clicked(self):
        type=str(self.secrOnContactCellTypeCB.currentText())
        currentText=str(self.secrOnContactLE.text())
        currentTypes=currentText.split(',')
        if currentText!="":
            if type not in currentTypes:
                self.secrOnContactLE.setText(currentText+","+type)
        else:
            self.secrOnContactLE.setText(type)

    @pyqtSignature("") # signature of the signal emited by the button           
    def on_secrAddRowPB_clicked(self):
        field=string.rstrip(str(self.secrFieldCB.currentText()))
        cellType=string.rstrip(str(self.secrCellTypeCB.currentText()))
        
        secrRate=0.0
        try:
            secrRate=float(str(self.secrRateLE.text()))
        except Exception:
            secrRate=0.0
                            
        secrOnContact=str(self.secrOnContactLE.text())
        
        secrType="uniform"
        if self.secrOnContactRB.isChecked():
            secrType="on contact"
        elif self.secrConstConcRB.isChecked():
            secrType="constant concentration"
            
        rows=self.secretionTable.rowCount()
        self.secretionTable.insertRow(rows)
                
        self.secretionTable.setItem(rows,0,QTableWidgetItem(field))
        self.secretionTable.setItem(rows,1,QTableWidgetItem(cellType))
        self.secretionTable.setItem(rows,2,QTableWidgetItem(str(secrRate)))
        self.secretionTable.setItem(rows,3,QTableWidgetItem(secrOnContact))        
        self.secretionTable.setItem(rows,4,QTableWidgetItem(str(secrType)))
                
        #reset entry lines        
        self.secrOnContactLE.setText('')            
        
    @pyqtSignature("") # signature of the signal emited by the button            
    def on_secrRemoveRowsPB_clicked(self):
    
        selectedItems=self.secretionTable.selectedItems()
        rowDict={}
        for item in selectedItems:
            rowDict[item.row()]=0
        rows=rowDict.keys()
        rows.sort()
        rowsSize=len(rows)
        
        
        for idx in range(rowsSize-1,-1,-1):
            
            row=rows[idx]
            self.secretionTable.removeRow(row)
            
    @pyqtSignature("") # signature of the signal emited by the button        
    def on_secrClearTablePB_clicked(self):
        # self.secretionTable.clear()
        rows=self.secretionTable.rowCount()
        for idx in range(rows-1,-1,-1):
            self.secretionTable.removeRow(idx)
        
            
    #CHEMOTAXIS        
    @pyqtSignature("bool") # signature of the signal emited by the button    
    def on_chemSatRB_toggled(self,_flag):        
        if _flag:           
            self.satCoefLB.setText("Saturation Coef.")
            self.satCoefLB.setHidden(False)
            self.satChemLE.setHidden(False)
        else:            
            self.satCoefLB.setHidden(True)
            self.satChemLE.setHidden(True)
            self.satChemLE.setText('')
            
            
    @pyqtSignature("bool") # signature of the signal emited by the radio button    
    def on_chemSatLinRB_toggled(self,_flag):
        if _flag:           
            self.satCoefLB.setText("Saturation Coef. Linear")
            self.satCoefLB.setHidden(False)
            self.satChemLE.setHidden(False)
        else:            
            self.satCoefLB.setHidden(True)
            self.satChemLE.setHidden(True)
            self.satChemLE.setText('')
            
            
    @pyqtSignature("") # signature of the signal emited by the button    
    def on_chemotaxTowardsPB_clicked(self):
        type=str(self.chemTowardsCellTypeCB.currentText())
        currentText=str(self.chemotaxTowardsLE.text())
        currentTypes=currentText.split(',')
        if currentText!="":
            if type not in currentTypes:
                self.chemotaxTowardsLE.setText(currentText+","+type)
        else:
            self.chemotaxTowardsLE.setText(type)
            
            
    @pyqtSignature("") # signature of the signal emited by the button           
    def on_chemotaxisAddRowPB_clicked(self):
        field=string.rstrip(str(self.chemFieldCB.currentText()))
        cellType=string.rstrip(str(self.chemCellTypeCB.currentText()))
        
        lambda_=0.0
        try:
            lambda_=float(str(self.lambdaChemLE.text()))
        except Exception:
            lambda_=0.0
            
        saturationCoef=0.0    
        if not self.chemRegRB.isChecked(): 
            try:
                saturationCoef=float(str(self.satChemLE.text()))
            except Exception:
                saturationCoef=0.0
                
        chemotaxTowardsTypes=str(self.chemotaxTowardsLE.text())
        
        chemotaxisType="regular"
        if self.chemSatRB.isChecked():
            chemotaxisType="saturation"
        elif self.chemSatLinRB.isChecked():
            chemotaxisType="saturation linear"
            
        rows=self.chamotaxisTable.rowCount()
        self.chamotaxisTable.insertRow(rows)
        
        
        self.chamotaxisTable.setItem(rows,0,QTableWidgetItem(field))
        self.chamotaxisTable.setItem(rows,1,QTableWidgetItem(cellType))
        self.chamotaxisTable.setItem(rows,2,QTableWidgetItem(str(lambda_)))
        self.chamotaxisTable.setItem(rows,3,QTableWidgetItem(chemotaxTowardsTypes))        
        self.chamotaxisTable.setItem(rows,4,QTableWidgetItem(str(saturationCoef)))
        self.chamotaxisTable.setItem(rows,5,QTableWidgetItem(chemotaxisType))
        
        #reset entry lines        
        self.chemotaxTowardsLE.setText('')
        
    @pyqtSignature("") # signature of the signal emited by the button            
    def on_chemotaxisRemoveRowsPB_clicked(self):
    
        selectedItems=self.chamotaxisTable.selectedItems()
        rowDict={}
        for item in selectedItems:
            rowDict[item.row()]=0
        rows=rowDict.keys()
        rows.sort()
        rowsSize=len(rows)
        print "rows=",rows
        
        for idx in range(rowsSize-1,-1,-1):
            
            row=rows[idx]
            self.chamotaxisTable.removeRow(row)
            
    @pyqtSignature("") # signature of the signal emited by the button        
    def on_chemotaxisClearTablePB_clicked(self):
        # self.chamotaxisTable.clear()
        rows=self.chamotaxisTable.rowCount()
        for idx in range(rows-1,-1,-1):
            self.chamotaxisTable.removeRow(idx)
        
    @pyqtSignature("") # signature of the signal emited by the button    
    def on_afMoleculeAddPB_clicked(self):
        
        molecule=str(self.afMoleculeLE.text())
        molecule=string.rstrip(molecule)
        rows=self.afTable.rowCount()
        if molecule =="":
            return

        # check if molecule with this name already exist               
        moleculeAlreadyExists=False
        for rowId in range(rows):
            name=str(self.afTable.item(rowId,0).text())
            name=string.rstrip(name)
            if name==molecule:
                moleculeAlreadyExists=True
                break
        
        if moleculeAlreadyExists:
            QMessageBox.warning(self,"Molecule Name Already Exists","Molecule name already exist. Please choose different name",QMessageBox.Ok)
            return
            
        self.afTable.insertRow(rows)        
        moleculeItem=QTableWidgetItem(molecule)
        self.afTable.setItem (rows,0,  moleculeItem)
        
        # reset molecule entry line
        self.afMoleculeLE.setText("")
        return 
        
    @pyqtSignature("") # signature of the signal emited by the button
    def on_clearAFTablePB_clicked(self):
        rows=self.afTable.rowCount()
        for i in range (rows-1,-1,-1):
            self.afTable.removeRow(i)
            
            
    @pyqtSignature("") # signature of the signal emited by the button    
    def on_cmcMoleculeAddPB_clicked(self):        
        cadherin=str(self.cmcMoleculeLE.text())
        cadherin=string.rstrip(cadherin)
        rows=self.cmcTable.rowCount()
        if cadherin =="":
            return
            
        # check if cadherin with this name already exist               
        cadherinAlreadyExists=False
        for rowId in range(rows):
            name=str(self.cmcTable.item(rowId,0).text())
            name=string.rstrip(name)
            if name==cadherin:
                cadherinAlreadyExists=True
                break
        
        if cadherinAlreadyExists:
            QMessageBox.warning(self,"Cadherin Name Already Exists","Cadherin name already exist. Please choose different name",QMessageBox.Ok)
            return
            
        self.cmcTable.insertRow(rows)        
        cadherinItem=QTableWidgetItem(cadherin)
        self.cmcTable.setItem (rows,0,  cadherinItem)
        
        # reset cadherin entry line
        self.cmcMoleculeLE.setText("")
        
        return 
    @pyqtSignature("") # signature of the signal emited by the button
    def on_clearCMCTablePB_clicked(self):
        rows=self.cmcTable.rowCount()
        for i in range (rows-1,-1,-1):
            self.cmcTable.removeRow(i)

    @pyqtSignature("") # signature of the signal emited by the button
    def on_plotAddPB_clicked(self):
        
        plotName=str(self.plotLE.text())
        plotName=string.rstrip(plotName)
        
        plotType=str(self.plotTypeCB.currentText())
        plotType=string.rstrip(plotType)

        if plotName =="":
            return
        
        # check if plot with this name already exist               
        rows=self.plotTable.rowCount()
        
        plotAlreadyExists=False
        for rowId in range(rows):
            name=str(self.plotTable.item(rowId,0).text())
            name=string.rstrip(name)
            if name==plotName:
                plotAlreadyExists=True
                break
        
        if plotAlreadyExists:
            QMessageBox.warning(self,"Plot name already exists","Plot name already exist. Please choose different name",QMessageBox.Ok)
            return
        
        
            
        self.plotTable.insertRow(rows)        
        plotNameItem=QTableWidgetItem(plotName)
        self.plotTable.setItem (rows,0,  plotNameItem)
        
        plotTypeItem=QTableWidgetItem(plotType)
        self.plotTable.setItem (rows,1,  plotTypeItem)
        
            
        
        # reset cell type entry line
        self.plotLE.setText("")
        return 

    @pyqtSignature("") # signature of the signal emited by the button
    def on_clearPlotTablePB_clicked(self):
        rows=self.plotTable.rowCount()
        for i in range (rows-1,-1,-1):
            self.plotTable.removeRow(i)

            
    @pyqtSignature("") # signature of the signal emited by the button        
    def on_dirPB_clicked(self):
        name=str(self.nameLE.text())
        name=string.rstrip(name)
        
        projDir=self.plugin.configuration.setting("RecentNewProjectDir")
        
        if name!="":
            dir=QFileDialog.getExistingDirectory(self,"Specify Location for your project",projDir)
            self.plugin.configuration.setSetting("RecentNewProjectDir",dir)
            self.dirLE.setText(dir)
        # else:
            # QMessageBox.information(self,"Project Name Missing","Please specify project name first",QMessageBox.Ok)
    
    # @pyqtSignature("QString") # signature of the signal emited by the button           
    # def on_latticeTypeCB_activated(self,_text):
        # if str(_text)=="Square":
            # self.connect2DCHB.setShown(True)
        # else:
            # self.connect2DCHB.setShown(False)
            # self.connect2DCHB.setChecked(False)
    
    
    # @pyqtSignature("") # signature of the signal emited by the button
    # def on_cellTypeLE_returnPressed(self):
        # print "GOT ENTER EVENT"
        # self.on_cellTypeAddPB_clicked()
        
    #setting up validators for the entry fields
    def setUpValidators(self):
        self.membraneFluctuationsLE.setValidator(QDoubleValidator())
        self.secrRateLE.setValidator(QDoubleValidator())
        self.lambdaChemLE.setValidator(QDoubleValidator())
        self.satChemLE.setValidator(QDoubleValidator())
        
    #initialize properties dialog
    def updateUi(self):
        self.setUpValidators()
        
        # Multi cad plugin is being deprecated
        self.contactMultiCadCHB.setEnabled(False)
    
        # have to set base size in QDesigner and then read it to rescale columns. For some reason reading size of the widget does not work properly
        pageIds=self.pageIds()        
        self.pageDict["FinalPage"]=[self.page(pageIds[-1]),len(pageIds)-1]
        self.pageDict["GeneralProperties"]=[self.page(1),1] 
        self.pageDict["CellType"]=[self.page(2),2] 
        self.pageDict["Diffusants"]=[self.page(3),3] 
        self.pageDict["Secretion"]=[self.page(5),5]           
        self.pageDict["Chemotaxis"]=[self.page(6),6]   
        self.pageDict["AdhesionFlex"]=[self.page(7),7]        
        self.pageDict["ContactMultiCad"]=[self.page(8),8]
        self.pageDict["PythonScript"]=[self.page(9),9]
        

        self.removePage(5)
        self.removePage(6)
        self.removePage(7)
        self.removePage(8)        
        
        self.nameLE.selectAll()
        projDir=self.plugin.configuration.setting("RecentNewProjectDir")
        print "projDir=",projDir
        if str(projDir)=="":
            projDir=os.environ["PREFIX_CC3D"]
        self.dirLE.setText(projDir)    
        
        # self.cellTypeLE.setFocus(True)
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
        #general properties page
        self.piffPB.setHidden(True)
        self.piffLE.setHidden(True)
        
        #chemotaxis page        
        baseSize=self.fieldTable.baseSize()
        self.fieldTable.setColumnWidth (0,baseSize.width()/2)
        self.fieldTable.setColumnWidth (1,baseSize.width()/2)
        self.fieldTable.horizontalHeader().setStretchLastSection(True)

        self.satCoefLB.setHidden(True)
        self.satChemLE.setHidden(True)
        
        #secretion page
        baseSize=self.secretionTable.baseSize()
        self.secretionTable.setColumnWidth (0,baseSize.width()/5)
        self.secretionTable.setColumnWidth (1,baseSize.width()/5)
        self.secretionTable.setColumnWidth (2,baseSize.width()/5)
        self.secretionTable.setColumnWidth (3,baseSize.width()/5)
        self.secretionTable.setColumnWidth (4,baseSize.width()/5)
        self.secretionTable.horizontalHeader().setStretchLastSection(True)
        
        
        self.secrAddOnContactPB.setHidden(True)
        self.secrOnContactCellTypeCB.setHidden(True)
        self.secrOnContactLE.setHidden(True)
        
        
        
        # AF molecule table
        self.afTable.horizontalHeader().setStretchLastSection(True)
        
        # CMC cadherin table
        self.cmcTable.horizontalHeader().setStretchLastSection(True)
        
        # plotTypeTable
        baseSize=self.plotTable.baseSize()
        self.plotTable.setColumnWidth (0,baseSize.width()/2)
        self.plotTable.setColumnWidth (1,baseSize.width()/2)
        self.plotTable.horizontalHeader().setStretchLastSection(True)

        
        # self.cellTypeTable.insertRow(0)
        # self.cellTypeTable.horizontalHeader().resizeSections(QHeaderView.Interactive)
        # self.cellTypeTable.horizontalHeader().setStretchLastSection(True)
        
        width=self.cellTypeTable.horizontalHeader().width()
        print "column 0 width=",self.cellTypeTable.horizontalHeader().sectionSize(0)
        print "column 1 width=",self.cellTypeTable.horizontalHeader().sectionSize(1)
        
        # width=self.cellTypeTable.width()
        print "size=",self.cellTypeTable.size()
        print "baseSize=",self.cellTypeTable.baseSize()
        
        print "width=",width
        print "column width=",self.cellTypeTable.columnWidth(0)
        # self.cellTypeTable.setColumnWidth (0,200/2)
        # self.cellTypeTable.setColumnWidth (1,200/2)
        # return
        
    def insertModulePage(self,_page):
        # get FinalPage id
        finalId=-1
        pageIds=self.pageIds()
        for id in pageIds:
            if self.page(id)==self.pageDict["FinalPage"]:
                finalId=id
                break
        if  finalId ==-1:
            print "COULD NOT INSERT PAGE  COULD NOT FIND LAST PAGE "
            return
        print "FinalId=",finalId
        
        self.setPage(finalId-1,_page)
        
    def removeModulePage(self,_page):
    
        pageIds=self.pageIds()
        for id in pageIds:
            if self.page(id)==_page:
                self.removePage(id)
                break
        
    def validateCurrentPage(self):
        
        print "THIS IS VALIDATE FOR PAGE ",self.currentId
        if self.currentId()==0:
            dir=str(self.dirLE.text())
            dir=string.rstrip(dir)
            name=str(self.nameLE.text())
            name=string.rstrip(name)
            if self.xmlRB.isChecked():
                self.removePage(self.pageDict["PythonScript"][1])
            else:
                self.setPage(self.pageDict["PythonScript"][1],self.pageDict["PythonScript"][0])
                
            if dir=="" or name=="":        
                QMessageBox.warning(self,"Missing information","Please specify name of the simulation and directory where it should be written to",QMessageBox.Ok)
                return False
            else:
                if dir!="":            
                    self.plugin.configuration.setSetting("RecentNewProjectDir",dir) 
                    print "CHECKING DIRECTORY "
                    #checking if directory is writeable
                    if not os.access(os.path.abspath(dir), os.W_OK):
                        print "CHECKING DIRECTORY "
                        QMessageBox.warning(self,"Write permission Error","You do not have write permissions to %s directory" %(os.path.abspath(dir)),QMessageBox.Ok)
                        return False
                    
        
            
                return True
        # general properties        
        if self.currentId()==1:
            if self.piffRB.isChecked() and string.rstrip(str(self.piffLE.text()))=='':
                QMessageBox.warning(self,"Missing information","Please specify name of the PIFF file",QMessageBox.Ok)                
                return False
        
            sim3DFlag=False
            
            if self.xDimSB.value()>1 and self.yDimSB.value()>1 and self.zDimSB.value()>1:
                sim3DFlag=True
            
            if sim3DFlag:
                self.lengthConstraintLocalFlexCHB.setChecked(False)
                self.lengthConstraintLocalFlexCHB.setShown(False)
                # self.connect2DCHB.setChecked(False)
                # self.connect2DCHB.setShown(False)
            else:   
                self.lengthConstraintLocalFlexCHB.setShown(True)                    
                # self.connect2DCHB.setShown(True)    
                
            if str(self.latticeTypeCB.currentText()) == "Square" and not sim3DFlag:
                self.connect2DCHB.setShown(True)
            else:   
                self.connect2DCHB.setShown(False)
                self.connect2DCHB.setChecked(False)
                
            
            return True
                
        if self.currentId()==2:
            # we only extract types from table here - it is not a validation strictly speaking
            
        # extract cell type information form the table
            self.typeTable=[]
            for row in range(self.cellTypeTable.rowCount()):
                type=str(self.cellTypeTable.item(row,0).text())
                freeze=False
                if self.cellTypeTable.item(row,1).checkState()==Qt.Checked:
                    print "self.cellTypeTable.item(row,1).checkState()=",self.cellTypeTable.item(row,1).checkState()
                    freeze=True
                self.typeTable.append([type,freeze])
                
            return True    
            
        if self.currentId()==3:
            # we only extract diffusants from table here - it is not a validation strictly speaking
            
        # extract diffusants information form the table
            self.diffusantDict={}
            for row in range(self.fieldTable.rowCount()):
                field=str(self.fieldTable.item(row,0).text())
                solver=str(self.fieldTable.item(row,1).text())
                try:
                    self.diffusantDict[solver].append(field)
                except LookupError:
                    self.diffusantDict[solver]=[field]
                
            # at this point we can fill all the cell types and fields widgets on subsequent pages
        
            self.chemCellTypeCB.clear()
            self.chemTowardsCellTypeCB.clear()
            self.chemFieldCB.clear()
            
            print "Clearing Combo boxes"

                
            for cellTypeTuple in self.typeTable: 
                if str(cellTypeTuple[0])!="Medium":
                    self.chemCellTypeCB.addItem(cellTypeTuple[0])                    
                self.chemTowardsCellTypeCB.addItem(cellTypeTuple[0])
                
            for solverName,fields in self.diffusantDict.iteritems():    
                for fieldName in fields:
                    self.chemFieldCB.addItem(fieldName)
               
            
            # secretion plugin
            self.secrFieldCB.clear()
            self.secrCellTypeCB.clear()
            self.secrOnContactCellTypeCB.clear()
            
            for cellTypeTuple in self.typeTable: 
                self.secrCellTypeCB.addItem(cellTypeTuple[0])
                self.secrOnContactCellTypeCB.addItem(cellTypeTuple[0])
            
            for solverName,fields in self.diffusantDict.iteritems():    
                for fieldName in fields:
                    self.secrFieldCB.addItem(fieldName)
            
            return True 
            
        if self.currentId()==4:
            print self.pageDict
            if self.secretionCHB.isChecked():
                self.setPage(self.pageDict["Secretion"][1],self.pageDict["Secretion"][0])
            else:
                self.removePage(self.pageDict["Secretion"][1])
            
            if self.chemotaxisCHB.isChecked():
                self.setPage(self.pageDict["Chemotaxis"][1],self.pageDict["Chemotaxis"][0])
            else:
                self.removePage(self.pageDict["Chemotaxis"][1])
            
            if self.contactMultiCadCHB.isChecked():
                self.setPage(self.pageDict["ContactMultiCad"][1],self.pageDict["ContactMultiCad"][0])
            else:
                self.removePage(self.pageDict["ContactMultiCad"][1])
        
            if self.adhesionFlexCHB.isChecked():
                self.setPage(self.pageDict["AdhesionFlex"][1],self.pageDict["AdhesionFlex"][0])
            else:
                self.removePage(self.pageDict["AdhesionFlex"][1])
                # self.removeModulePage(self.pageDict["AdhesionFlex"])            
                # pageIds=self.pageIds()
                # for id in pageIds:
                    # if self.page(id)==self.pageDict["AdhesionFlex"]:
                        # self.removePage(id)

            
            return True
            
        if self.currentPage() == self.pageDict["ContactMultiCad"][0]:  
            if not self.cmcTable.rowCount():
                QMessageBox.warning(self,"Missing information","Please specify at least one cadherin name to be used in ContactMultiCad plugin",QMessageBox.Ok)
                return False
            else:
                return True
            
        if self.currentPage() == self.pageDict["AdhesionFlex"][0]:  
            if not self.afTable.rowCount():
                QMessageBox.warning(self,"Missing information","Please specify at least one adhesion molecule name to be used in AdhesionFlex plugin",QMessageBox.Ok)
                return False
            else:
                return True
            
            
        return True
        
    def makeProjectDirectories(self,dir,name):
        
        try:
            self.mainProjDir=os.path.join(dir,name)
            self.plugin.makeDirectory(self.mainProjDir)
            self.simulationFilesDir=os.path.join(self.mainProjDir,"Simulation")
            self.plugin.makeDirectory(self.simulationFilesDir)
            
        except IOError,e:
            raise IOError
        return
        
    def generateNewProject(self):    
        dir=str(self.dirLE.text())
        dir=os.path.abspath(dir)
        dir=string.rstrip(dir)
        name=str(self.nameLE.text())        
        name=string.rstrip(name)
        
        print "name=",name," dir=",dir
        self.makeProjectDirectories(dir,name)
        
        self.generalPropertiesDict={}
        self.generalPropertiesDict["Dim"]=[self.xDimSB.value(),self.yDimSB.value(),self.zDimSB.value()]
        self.generalPropertiesDict["MembraneFluctuations"]=float(str(self.membraneFluctuationsLE.text()))
        self.generalPropertiesDict["NeighborOrder"]=self.neighborOrderSB.value()
        self.generalPropertiesDict["MCS"]=self.mcsSB.value()
        self.generalPropertiesDict["LatticeType"]=str(self.latticeTypeCB.currentText())
        self.generalPropertiesDict["SimulationName"]=name
        
        self.generalPropertiesDict["Initializer"]=["uniform",None]
        if self.blobRB.isChecked():
            self.generalPropertiesDict["Initializer"]=["blob",None]
        elif self.piffRB.isChecked():
            piffPath=str(self.piffLE.text())
            piffPath=string.rstrip(piffPath)
            self.generalPropertiesDict["Initializer"]=["piff",piffPath]
            # trying to copy piff file into simulation dir of the project directory
            import shutil
            try:
                
                shutil.copy(piffPath,self.simulationFilesDir)
                basePiffPath=os.path.basename(piffPath)
                relativePiffPath=os.path.join(self.simulationFilesDir,basePiffPath)
                self.generalPropertiesDict["Initializer"][1]=self.getRelativePathWRTProjectDir(relativePiffPath)
                print "relativePathOF PIFF=",self.generalPropertiesDict["Initializer"][1]
                
            
            except shutil.Error, e:
                QMessageBox.warning(self,"Cannot copy PIFF file","Cannot copy PIFF file into project directory. Please check if the file exists and that you have necessary write permissions",QMessageBox.Ok)
                pass # ignore any copy errors        
                
            except IOError,e:
                QMessageBox.warning(self,"IO Error",e.__str__(),QMessageBox.Ok)
                pass
                
                
        self.cellTypeData={}
        
        # extract cell type information form the table
        for row in range(self.cellTypeTable.rowCount()):
            type=str(self.cellTypeTable.item(row,0).text())
            freeze=False
            if self.cellTypeTable.item(row,1).checkState()==Qt.Checked:
                print "self.cellTypeTable.item(row,1).checkState()=",self.cellTypeTable.item(row,1).checkState()
                freeze=True
            self.cellTypeData[row]=[type,freeze]    
            # self.typeTable.append([type,freeze])
        
        self.afData={}
        for row in range(self.afTable.rowCount()):
            molecule=str(self.afTable.item(row,0).text())
            self.afData[row]=molecule    
            
            
        self.afFormula=str(self.bindingFormulaLE.text())
        self.afFormula=string.rstrip(self.afFormula)            
        
        
        cmcTable=[]
        for row in range(self.cmcTable.rowCount()):
            cadherin=str(self.cmcTable.item(row,0).text())
            cmcTable.append(cadherin)
        
        plotTypeTable=[]
        for row in range(self.plotTable.rowCount()):
            plotName=str(self.plotTable.item(row,0).text())
            plotName=string.rstrip(plotName)
            
            plotType=str(self.plotTable.item(row,1).text())
            plotType=string.rstrip(plotType)
            plotTypeTable.append([plotName,plotType])
            
        self.pdeFieldData={}
        for row in range(self.fieldTable.rowCount()):
            chemFieldName=str(self.fieldTable.item(row,0).text())
            solverName=str(self.fieldTable.item(row,1).text())
            # chemFieldsTable.append([chemFieldName,solverName])
            self.pdeFieldData[chemFieldName]=solverName
        
            
        self.secretionData={} #format {field:[secrDict1,secrDict2,...]}      
        for row in range(self.secretionTable.rowCount()):
            secrFieldName=str(self.secretionTable.item(row,0).text())
            cellType=str(self.secretionTable.item(row,1).text())                        
            rate=0.0
            try:
                rate=float(str(self.secretionTable.item(row,2).text()))
            except Exception:
                rate=0.0            
                
            onContactWith=str(self.secretionTable.item(row,3).text())    
            
            
            secretionType=str(self.secretionTable.item(row,4).text())    
            
            secrDict={}
            secrDict["CellType"]=cellType
            secrDict["Rate"]=rate
            secrDict["OnContactWith"]=onContactWith
            
            secrDict["SecretionType"]=secretionType
            
            try:
                self.secretionData[secrFieldName].append(secrDict)
            except LookupError:
                self.secretionData[secrFieldName]=[secrDict]
        
        self.chemotaxisData={} #format {field:[chemDict1,chemDict2,...]}   
        for row in range(self.chamotaxisTable.rowCount()):
            chemFieldName=str(self.chamotaxisTable.item(row,0).text())
            cellType=str(self.chamotaxisTable.item(row,1).text())                        
            lambda_=0.0
            try:
                lambda_=float(str(self.chamotaxisTable.item(row,2).text()))
            except Exception:
                lambda_=0.0            
                
            chemotaxTowards=str(self.chamotaxisTable.item(row,3).text())    
            
            satCoef=0.0
            try:
                satCoef=float(str(self.chamotaxisTable.item(row,4).text()))
            except Exception:
                satCoef=0.0            
            
            chemotaxisType=str(self.chamotaxisTable.item(row,5).text())    
            
            chemDict={}
            chemDict["CellType"]=cellType
            chemDict["Lambda"]=lambda_
            chemDict["ChemotaxTowards"]=chemotaxTowards
            chemDict["SatCoef"]=satCoef
            chemDict["ChemotaxisType"]=chemotaxisType
            
            try:
                self.chemotaxisData[chemFieldName].append(chemDict)
            except LookupError:
                self.chemotaxisData[chemFieldName]=[chemDict]
            
            
        # constructing Project XMl Element        
        
        from XMLUtils import ElementCC3D
        simulationElement=ElementCC3D("Simulation",{"version":"3.6.2"})
        
        from CC3DMLGenerator.CC3DMLGeneratorBase import CC3DMLGeneratorBase
        xmlGenerator=CC3DMLGeneratorBase(self.simulationFilesDir,name)
        
        self.generateXML(xmlGenerator)
        # simulationElement.ElementCC3D("XMLScript",{"Type":"XMLScript"},self.getRelativePathWRTProjectDir(xmlGenerator.fileName))
        #end of generate XML ------------------------------------------------------------------------------------
        
            
        if self.pythonXMLRB.isChecked() or self.xmlRB.isChecked():
            xmlFileName=os.path.join(self.simulationFilesDir,name+".xml")
            xmlGenerator.saveCC3DXML(xmlFileName)
            simulationElement.ElementCC3D("XMLScript",{"Type":"XMLScript"},self.getRelativePathWRTProjectDir(xmlFileName))
            #end of generate XML ------------------------------------------------------------------------------------
            
            
        if self.pythonXMLRB.isChecked() or self.pythonOnlyRB.isChecked():
            #generate Python ------------------------------------------------------------------------------------        
            from CC3DPythonGenerator import CC3DPythonGenerator

            pythonGenerator=CC3DPythonGenerator(xmlGenerator)
            pythonGenerator.setPythonOnlyFlag(self.pythonOnlyRB.isChecked())
            
            

            pythonGenerator.setPlotTypeTable(plotTypeTable)
            
            if self.dictCB.isChecked():
                pythonGenerator.attachDictionary=True
            if self.listCB.isChecked():
                pythonGenerator.attachList=True
                
            # self.generatePythonConfigureSim(pythonGenerator)    
            
            self.generateSteppablesCode(pythonGenerator)        
            # before calling generateMainPythonScript we have to call generateSteppablesCode which generates also steppable registration lines
            pythonGenerator.generateMainPythonScript()
            simulationElement.ElementCC3D("PythonScript",{"Type":"PythonScript"},self.getRelativePathWRTProjectDir(pythonGenerator.mainPythonFileName))            
    
            simulationElement.ElementCC3D("Resource",{"Type":"Python"},self.getRelativePathWRTProjectDir(pythonGenerator.steppablesPythonFileName))            
            #end of generate Python ------------------------------------------------------------------------------------
            
        #including PIFFile in the .cc3d project description
        if self.generalPropertiesDict["Initializer"][0]=="piff":
            simulationElement.ElementCC3D("PIFFile",{},self.generalPropertiesDict["Initializer"][1])
        
        # save Project file
        projFileName=os.path.join(self.mainProjDir,name+".cc3d")
        simulationElement.CC3DXMLElement.saveXML(projFileName)
        #open newly created project in the ProjectEditor
        self.plugin.openCC3Dproject(projFileName)
        
    def generateSteppablesCode(self,pythonGenerator):
        if self.growthCHB.isChecked():
            pythonGenerator.generateGrowthSteppable()
        if self.mitosisCHB.isChecked():
            pythonGenerator.generateMitosisSteppable()    
        if self.deathCHB.isChecked():
            pythonGenerator.generateDeathSteppable()
            
        pythonGenerator.generateVisPlotSteppables()        
        
        pythonGenerator.generateSteppablePythonScript()
        pythonGenerator.generateSteppableRegistrationLines()
        pythonGenerator.generatePlotSteppableRegistrationLines()        
        
        
    def generateXML(self,generator):

        cellTypeDict=self.cellTypeData
        
        args=[]
        kwds={}
        kwds['insert_root_element']=generator.cc3d
        kwds['data']=cellTypeDict
        kwds['generalPropertiesData']=self.generalPropertiesDict
        
        kwds['afData']=self.afData
        kwds['formula']=self.afFormula
        kwds['chemotaxisData']=self.chemotaxisData
        kwds['pdeFieldData']=self.pdeFieldData
        kwds['secretionData']=self.secretionData
        
        

             
        
        generator.generatePottsSection(*args,**kwds)        
        
        generator.generateCellTypePlugin(*args,**kwds)
        
        if self.volumeFlexCHB.isChecked():
            generator.generateVolumeFlexPlugin(*args,**kwds) 
        if self.surfaceFlexCHB.isChecked():
            generator.generateSurfaceFlexPlugin(*args,**kwds)
        if self.volumeLocalFlexCHB.isChecked():
            generator.generateVolumeLocalFlexPlugin(*args,**kwds)            
        if self.surfaceLocalFlexCHB.isChecked():
            generator.generateSurfaceLocalFlexPlugin(*args,**kwds)            
            
        if self.extPotCHB.isChecked():
            generator.generateExternalPotentialPlugin(*args,**kwds)            
            
        if self.extPotLocalFlexCHB.isChecked():
            generator.generateExternalPotentialLocalFlexPlugin(*args,**kwds)           
            
        if self.comCHB.isChecked():
            generator.generateCenterOfMassPlugin(*args,**kwds)
        if self.neighborCHB.isChecked():
            generator.generateNeighborTrackerPlugin(*args,**kwds)
        if self.momentOfInertiaCHB.isChecked():
            generator.generateMomentOfInertiaPlugin(*args,**kwds)
        if self.pixelTrackerCHB.isChecked():
            generator.generatePixelTrackerPlugin(*args,**kwds)
            
        if self.boundaryPixelTrackerCHB.isChecked():
            generator.generateBoundaryPixelTrackerPlugin(*args,**kwds)
            
        if self.contactCHB.isChecked():
            generator.generateContactPlugin(*args,**kwds)
            
            
        if self.compartmentCHB.isChecked():        
            generator.generateCompartmentPlugin(*args,**kwds)
            
        if self.internalContactCB.isChecked():
            generator.generateContactInternalPlugin(*args,**kwds) 
            
        if self.contactLocalProductCHB.isChecked():
            generator.generateContactLocalProductPlugin(*args,**kwds)            
            
        if self.fppCHB.isChecked():
            generator.generateFocalPointPlasticityPlugin(*args,**kwds)            
            
        if self.elasticityCHB.isChecked():
            generator.generateElasticityTrackerPlugin(*args,**kwds)                    
            generator.generateElasticityPlugin(*args,**kwds)            
            
        if self.adhesionFlexCHB.isChecked():
            generator.generateAdhesionFlexPlugin(*args,**kwds)
             
        if self.chemotaxisCHB.isChecked():
            generator.generateChemotaxisPlugin(*args,**kwds)

        if self.lengthConstraintCHB.isChecked():            
            generator.generateLengthConstraintPlugin(*args,**kwds)
            
        if self.lengthConstraintLocalFlexCHB.isChecked():            
            generator.generateLengthConstraintLocalFlexPlugin(*args,**kwds)
            
            
        if self.connectGlobalCHB.isChecked():            
            generator.generateConnectivityGlobalPlugin(*args,**kwds)            
            
        if self.connectGlobalByIdCHB.isChecked():            
            generator.generateConnectivityGlobalByIdPlugin(*args,**kwds)            

            
        if self.connect2DCHB.isChecked():            
            generator.generateConnectivityPlugin(*args,**kwds)
            
            
        if self.secretionCHB.isChecked():
            generator.generateSecretionPlugin(*args,**kwds)
            
        # if self.pdeSolverCallerCHB.isChecked():
            # xmlGenerator.generatePDESolverCaller()

            
            
        #PDE solvers 
        #getting a list of solvers to be generated
        list_of_solvers=self.diffusantDict.keys()
        for solver in list_of_solvers:
            solverGeneratorFcn=getattr(generator,'generate'+solver)
            solverGeneratorFcn(*args,**kwds)
        # if self.fieldTable.rowCount():           
            # generator.generateDiffusionSolverFE(*args,**kwds)            
            # generator.generateFlexibleDiffusionSolverFE(*args,**kwds)            
            # generator.generateFastDiffusionSolver2DFE(*args,**kwds)            
            # generator.generateKernelDiffusionSolver(*args,**kwds)            
            # generator.generateSteadyStateDiffusionSolver(*args,**kwds)            
            
            
        if  self.boxWatcherCHB.isChecked():
            generator.generateBoxWatcherSteppable(*args,**kwds)
            
        #cell layout initializer    
        if self.uniformRB.isChecked():
            generator.generateUniformInitializerSteppable(*args,**kwds)            
        elif self.blobRB.isChecked():
            generator.generateBlobInitializerSteppable(*args,**kwds)           
        elif self.piffRB.isChecked():
            generator.generatePIFInitializerSteppable(*args,**kwds)
        if  self.pifDumperCHB.isChecked():
            generator.generatePIFDumperSteppable(*args,**kwds)
        
        
        # generator.generatePottsSection(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)        
        
        # generator.generateCellTypePlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
        
        # if self.volumeFlexCHB.isChecked():
            # generator.generateVolumeFlexPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict) 
        # if self.surfaceFlexCHB.isChecked():
            # generator.generateSurfaceFlexPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
        # if self.volumeLocalFlexCHB.isChecked():
            # generator.generateVolumeLocalFlexPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)            
        # if self.surfaceLocalFlexCHB.isChecked():
            # generator.generateSurfaceLocalFlexPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)            
            
        # if self.extPotCHB.isChecked():
            # generator.generateExternalPotentialPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)            
            
        # if self.extPotLocalFlexCHB.isChecked():
            # generator.generateExternalPotentialLocalFlexPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)           
            
        # if self.comCHB.isChecked():
            # generator.generateCenterOfMassPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
        # if self.neighborCHB.isChecked():
            # generator.generateNeighborTrackerPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
        # if self.momentOfInertiaCHB.isChecked():
            # generator.generateMomentOfInertiaPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
        # if self.pixelTrackerCHB.isChecked():
            # generator.generatePixelTrackerPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
            
        # if self.boundaryPixelTrackerCHB.isChecked():
            # generator.generateBoundaryPixelTrackerPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
            
        # if self.contactCHB.isChecked():
            # generator.generateContactPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict)
            
            
        # if self.compartmentCHB.isChecked():        
            # generator.generateCompartmentPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict)
            
        # if self.internalContactCB.isChecked():
            # generator.generateContactInternalPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict) 
            
        # if self.contactLocalProductCHB.isChecked():
            # generator.generateContactLocalProductPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict)            
            
        # if self.fppCHB.isChecked():
            # generator.generateFocalPointPlasticityPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict)            
            
        # if self.elasticityCHB.isChecked():
            # generator.generateElasticityTrackerPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict)                    
            # generator.generateElasticityPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict)            
            
        # # if self.contactMultiCadCHB.isChecked():
            # # xmlGenerator.generateContactMultiCadPlugin()
            
        # if self.adhesionFlexCHB.isChecked():
            # generator.generateAdhesionFlexPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,afData=self.afData,formula=self.afFormula)
             
        # if self.chemotaxisCHB.isChecked():
            # generator.generateChemotaxisPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict, chemotaxisData=self.chemotaxisData, pdeFieldData=self.pdeFieldData)

        # if self.lengthConstraintCHB.isChecked():            
            # generator.generateLengthConstraintPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
            
        # if self.lengthConstraintLocalFlexCHB.isChecked():            
            # generator.generateLengthConstraintLocalFlexPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
            
            
        # if self.connectGlobalCHB.isChecked():            
            # generator.generateConnectivityGlobalPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)            
            
        # if self.connectGlobalByIdCHB.isChecked():            
            # generator.generateConnectivityGlobalByIdPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)            

            
        # if self.connect2DCHB.isChecked():            
            # generator.generateConnectivityPlugin(insert_root_element=xmlGenerator.cc3d)
            
            
        # if self.secretionCHB.isChecked():
            # generator.generateSecretionPlugin(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict, secretionData=self.secretionData)
            
        # # if self.pdeSolverCallerCHB.isChecked():
            # # xmlGenerator.generatePDESolverCaller()

            
            
        # #PDE solvers 
        # if self.fieldTable.rowCount():
            # # generator.generatePDESolvers(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict, secretionData=self.secretionData,generalPropertiesData=self.generalPropertiesDict,pdeFieldData=self.pdeFieldData)
            # generator.generateFlexibleDiffusionSolverFE(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict, secretionData=self.secretionData,generalPropertiesData=self.generalPropertiesDict,pdeFieldData=self.pdeFieldData)            
            
        # if  self.boxWatcherCHB.isChecked():
            # generator.generateBoxWatcherSteppable(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
            
        # #cell layout initializer    
        # if self.uniformRB.isChecked():
            # generator.generateUniformInitializerSteppable(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)            
        # elif self.blobRB.isChecked():
            # generator.generateBlobInitializerSteppable(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)           
        # elif self.piffRB.isChecked():
            # generator.generatePIFInitializerSteppable(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
        # if  self.pifDumperCHB.isChecked():
            # generator.generatePIFDumperSteppable(insert_root_element=xmlGenerator.cc3d, data=cellTypeDict,generalPropertiesData=self.generalPropertiesDict)
            
        
        # xmlGenerator.saveCC3DXML()
        #end of generate XML ------------------------------------------------------------------------------------
        
    def findRelativePathSegments(self,basePath,p, rest=[]):
    
        """
            This function finds relative path segments of path p with respect to base path    
            It returns list of relative path segments and flag whether operation succeeded or not    
        """
        
        h,t = os.path.split(p)
        pathMatch=False
        if h==basePath:
            pathMatch=True
            return [t]+rest,pathMatch
        print "(h,t,pathMatch)=",(h,t,pathMatch)
        if len(h) < 1: return [t]+rest,pathMatch
        if len(t) < 1: return [h]+rest,pathMatch
        return self.findRelativePathSegments(basePath,h,[t]+rest)
        
    def findRelativePath(self,basePath,p):
        relativePathSegments,pathMatch=self.findRelativePathSegments(basePath,p)
        if pathMatch:
            relativePath=""
            for i in range(len(relativePathSegments)):
                segment=relativePathSegments[i]
                relativePath+=segment
                if i !=len(relativePathSegments)-1:
                    relativePath+="/" # we use unix style separators - they work on all (3) platforms
            return relativePath
        else:
            return p   
            
    def getRelativePathWRTProjectDir(self,path):
        return self.findRelativePath(self.mainProjDir,path)
             
    
        