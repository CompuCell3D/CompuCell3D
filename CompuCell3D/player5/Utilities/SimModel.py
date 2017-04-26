
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtXml import *
from PyQt5.QtWidgets import *

from Utilities.TreeMapper import *
import copy

PROPERTY, VALUE = range(2)

# The SimModel is specific to the CompuCell3D XML simulation models!

class SimModel(QAbstractItemModel):
    #class SimModel creates model for QTreeView
    
    def __init__(self, domDoc, parent=None):
        #Constructor
        #@param domDoc refers to the QDomDocument

        QAbstractItemModel.__init__(self, parent)
        
        # Store the QDomDocument (root of the document tree) in the private attribute
        self.__domDoc   = domDoc
        self.__printFlag=False

        
        self.__rootItem = treeNode(domDoc)
        
        if self.__rootItem:
            print "THIS IS ROOT ITEM=",self.__rootItem.name()
            print "ROOT ITEM DOMNode=",self.__rootItem.domNode().getName()
            self.checkSanity()
            # text = raw_input('Enter text here->')
            
        self.__isDirty      = False
        self.__dirtyModules={}
        self.__headers  = ["Property", "Value"]
    
    def setPrintFlag(self,_flag):
        self.__printFlag=_flag
        
    def addDirtyModule(self,_moduleCategory,name):
        if self.__dirtyModules.has_key(_moduleCategory):
            self.__dirtyModules[_moduleCategory][name]=0
        else:
            self.__dirtyModules[_moduleCategory]={name:0}
    def getDirtyModules(self):
        return self.__dirtyModules
    
    def columnCount(self, parent): #interface: done
        # if parent.isValid():
            # return parent.internalPointer().columnCount()
        # else:
            # return self.__rootItem.columnCount()
        return len(self.__headers)
        
    def treeItemFromIndex(self,_itemIndex):
        if _itemIndex.isValid():
            return _itemIndex.internalPointer()
        else:
            return self.__rootItem
            
    def data(self, index, role = Qt.DisplayRole): #interface: done
        if role != Qt.DisplayRole or not index.isValid():
            return QVariant()

        node = index.internalPointer()
        rowdata = [node.name(), node.value()]
        
        # Specify which data to display in each column!
        if index.column() in (PROPERTY, VALUE):
            return QVariant(rowdata[index.column()])
        
        return QVariant()

    
    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        
        if index.column() == VALUE:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role): #interface: done
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if 0 <= section <= len(self.__headers):
                return QVariant(self.__headers[section])

        return QVariant()

    def index(self, row, column, parentIndex): #interface: done
        if self.__rootItem is None:
            return QModelIndex()
        parentTreeItem=self.treeItemFromIndex(parentIndex)
        return self.createIndex(row,column,parentTreeItem.child(row))
        
        # if self.__rootItem is None:
            # return QModelIndex()
        # elif not parentIndex.isValid():     # Root item doesn't have parent
            # # print " row=",row," column=",column," index is invalid"
            # parentItem = self.__rootItem    # Object of TreeItem
        # else:
            # parentItem = parentIndex.internalPointer()
            
            
            
        # if 0 <= row and row < parentItem.childCount():
            # childItem = parentItem.child(row)
            # # '#text' node is not displayed 
            # if (childItem is not None):
                # # print " row=",row," column=",column," parent Item name=",parentItem.domNode().name
                # # print "self.__printFlag=",self.__printFlag
                # # if self.__printFlag:
                    # # print "child Item=",childItem.name()
                # return self.createIndex(row, column, childItem)

        # return QModelIndex()
        
    def parent(self, childIndex): #interface: done
        if not childIndex.isValid():
            return QModelIndex()
        
        childItem = childIndex.internalPointer()
        parentItem = childItem.parent()
        
        # if (parentItem is None) or (parentItem == self.__rootItem):
        if parentItem is None:
            return QModelIndex()
            
        grandparentItem=parentItem.parent()
        if grandparentItem is None:
            return QModelIndex()
        # row=grandparentItem.
        # return self.createIndex(row, childIndex.column(), parentItem)
        
        index=self.createIndex(parentItem.siblingIdx(), 0, parentItem)
        # print "parentItem=",parentItem.name()
        return index
    
        # if not childIndex.isValid():
            # return QModelIndex()
        
        # childItem = childIndex.internalPointer()
        # parentItem = childItem.parent()
        
        # # if (parentItem is None) or (parentItem == self.__rootItem):
        # if (parentItem is None) or (parentItem == self.__rootItem):
            # return QModelIndex()

        # return self.createIndex(parentItem.siblingIdx(), 0, parentItem)

    def rowCount(self, parentIndex): #interface: done
        if self.__rootItem is None:
            return 0
        elif not parentIndex.isValid():
            parentItem = self.__rootItem
        else:
            parentItem = parentIndex.internalPointer()

        return parentItem.childCount()
    
    # Sets the data of the tree node. 
    # (Fix it?) Even if the data has not been changed it still sets self.__isDirty = True

    def conv_to_int(self,obj):
        try:
            int_obj = int(obj)
            return int_obj, True
        except ValueError:
            return None, False

    def conv_to_float(self,obj):
        try:
            float_obj = float(obj)
            return float_obj, True
        except ValueError:
            return None, False

    def setData(self, index, value, role=Qt.EditRole):

        if index.isValid() and 0 <= index.row() < self.rowCount(index.parent()):
            column = index.column()

            if column == VALUE:  # and index.model().data(index).toString() != "":
                # Sets the value of the node
                item = index.internalPointer()

                # Check if edited value preserves type
                # str = value.toString()
                str = value.value()

                Ival, Iok = self.conv_to_int(str)
                Fval, Fok = self.conv_to_float(str)

                # Ival, Iok = str.toInt()
                # Fval, Fok = str.toFloat()
                if (Iok and item.type() == "int") \
                        or (Fok and item.type() == "float" and not Iok) \
                        or (item.type() == "string" and not Iok and not Fok):
                    item.setValue(str)
                    self.addDirtyModule(item.getSuperParent().name(), item.getSuperParent().value())
                    print "dirty modules=", self.__dirtyModules
                    print "item SuperParent type=", item.getSuperParent().name(), " type=", item.getSuperParent().value()
                    print "item.domNode()=", item.domNode()
                    print "item.domNode().getName()=", item.domNode().getName()
                    print "item.domNode().getText()=", item.domNode().getText()
                    # item.domNode().setNodeValue(str) #setNodeValue()

            self.__isDirty = True  # Variable that chacks if data has been modified!
            self.dataChanged.emit(index, index)
            # self.emit(SIGNAL("dataChanged(QModelIndex,QModelIndex)"), index, index)
            # self.emit(SIGNAL("dataChanged(QModelIndex,QModelIndex)"), index, index)
            return True

        return False

    # def setData(self, index, value, role=Qt.EditRole):
    #
    #     if index.isValid() and 0 <= index.row() < self.rowCount(index.parent()):
    #         column = index.column()
    #
    #         if column == VALUE: # and index.model().data(index).toString() != "":
    #             # Sets the value of the node
    #             item = index.internalPointer()
    #
    #             # Check if edited value preserves type
    #             str = value.toString()
    #
    #             Ival, Iok = str.toInt()
    #             Fval, Fok = str.toFloat()
    #             if (Iok and item.type() == "int") \
    #                 or (Fok and item.type() == "float" and not Iok) \
    #                 or (item.type() == "string" and not Iok and not Fok):
    #                 item.setValue(str)
    #                 self.addDirtyModule(item.getSuperParent().name(),item.getSuperParent().value())
    #                 print "dirty modules=",self.__dirtyModules
    #                 print "item SuperParent type=",item.getSuperParent().name()," type=",item.getSuperParent().value()
    #                 print "item.domNode()=",item.domNode()
    #                 print "item.domNode().getName()=",item.domNode().getName()
    #                 print "item.domNode().getText()=",item.domNode().getText()
    #                 # item.domNode().setNodeValue(str) #setNodeValue()
    #
    #         self.__isDirty = True # Variable that chacks if data has been modified!
    #         self.emit(SIGNAL("dataChanged(QModelIndex,QModelIndex)"), index, index)
    #         return True
    #
    #     return False
     
    def domDocIsDirty(self):
        return self.__isDirty
    
    def setDirty(self, dirty):
        self.__isDirty = dirty
        
    def domDoc(self):
        return self.__domDoc
           
    def checkSanity(self):pass
#    def checkSanity(self):
#        print "rootTI=",self.__rootItem.name," self.__rootItem.domNode().getName()=",self.__rootItem.domNode().getName()
#        pottsIdx=self.createIndex(0,0,self.__rootItem.child(0))
#        print "pottsIdx.internalPointer().name=",pottsIdx.internalPointer().name()
#        
#        volumeIdx=self.createIndex(1,0,self.__rootItem.child(1))
#        print "volumeIdx.internalPointer().value()=",volumeIdx.internalPointer().value()
#        
#        tempIdx=self.createIndex(3,0,pottsIdx.internalPointer().child(3))
#        print "tempIdx.internalPointer().name()=",tempIdx.internalPointer().name()
#        print "tempIdx.internalPointer()=",tempIdx.internalPointer()
#        print "tempIdx.internalPointer().domNode().getName()=",tempIdx.internalPointer().domNode().getName()
#        
#        print "CHECKING TREE SANITY"
#        pottsItem=self.__rootItem.child(0)
#        print "pottsItem=",pottsItem.name()," pottsItem.domNode().getName()=",pottsItem.domNode().getName()
#        
#        # print "CHECKING TREE SANITY"
 