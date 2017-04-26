
from PyQt5.QtCore import *
from PyQt5.QtXml import *

MODULENAME='TreeMapper.py:--'

class TreeItem:
    def __init__(self, name, value=""):
        self.__itemName     = name
        self.__itemValue    = value # The type of self.__itemValue is string 
        
        # Holds the type of the value: "int", "float", "string" 
        self.__type         = None
        # Holds the type of element: "attribute", "element"
        self.__elementType="element"
        
        self.__childItems   = []
        self.__parentItem   = None # By default parent item is None (as root item)!
        
        # setting superParents -they are either Potts, Metadata, Plugin or Steppable. SuperParents are used in steering. Essentially when we change any TreeItem in the TreeView
        # we can quickly extract name of the superParent and tell CC3D to reinitialize module that superParent describes. Otherwise if we dont keep track of super parents we would either have to do:
        # 1. time consuming and messy tracking back of which module needs to be changed in response to change in one of the parameter
        # or
        # 2. reinitialize all modules each time a single parameter is changed
        
        self.__superParent = None
        
        # Very important! Holds the reference to the CC3DXMLElement that correspond to this TreeItem
        self.__domNode      = None 
        self.__treeItemId = None # this is essentially a C++ pointer to CC3DXMLElement converted to long. It is used to identify TreeItems. Otherwise differentiating between different treeItems would be less elegant
        
        self.setType(value)
    def getTreeItemId(self):
        return self.__treeItemId
    
    def setCC3DXMLElement(self,_cc3dXMLElement):
        self.__domNode=_cc3dXMLElement
        self.__treeItemId=_cc3dXMLElement.getPointerAsLong()
        
    def setElementType(self,_elementType):
        self.__elementType=_elementType
        
    def setSuperParent(self,_superParent):
        self.__superParent=_superParent
        
    def getSuperParent(self):
        return self.__superParent
    
    def parent(self):
        return self.__parentItem
    
    def addChild(self,  child):
        if child is not None:
            # Parent is set when the child is added!
            child.setParent(self)       
            self.__childItems.append(child)
            # self.__childItems[i] = child
        
    def removeChild(self, childName):
        # for i in range(self.childCount()):
            # if self.__childItems[i].name() == childName:
                # del self.__childItems[i]
        for i in self.__childItems:
            if self.__childItems[i].name() == childName:
                del self.__childItems[i]

    
    def child(self, i):
        # if 0 <= i < self.childCount():
            # return self.__childItems[i]
        # else:
            # return None
        if 0 <= i < len(self.__childItems):
            return self.__childItems[i]
        else:
            return None
        
    def childCount(self):
        return len(self.__childItems)
    
    def siblingIdx(self):
        if self.parent() is not None:
            for i in range(self.parent().childCount()):
                if self.parent().child(i).getTreeItemId() == self.getTreeItemId():
                    return i

        return 0
    
        # if self.parent() is not None:
            # for i in range(self.parent().childCount()):
                # if self.parent().child(i).name() == self.name():
                    # return i

        # return 0
    
    def hasChildItems(self):
        return self.childCount() != 0
    
    def firstChild(self):
        if self.hasChildItems():
            return self.__childItems[0]
        else:
            return None
    
    def setParent(self, parent):
        self.__parentItem = parent
        
    def hasParent(self):
        if self.__parentItem is not None:
            return True
        
        return False
    
    def setDomNode(self, domNode):
        self.__domNode = domNode
        
    def domNode(self):
        return self.__domNode
        
    def setName(self, name):
        self.__itemName = name
        
    def setItemValueOnly(self,value):
        self.__itemValue = value    
        
    def setValue(self, value):
    
        self.__itemValue = value
        if self.__elementType=="attribute":
            from XMLUtils import dictionaryToMapStrStr as d2mss
            self.__domNode.updateElementAttributes(d2mss({self.__itemName:str(self.__itemValue)}))
            # # # print MODULENAME,"UPDATING ATTRIBUTE ",self.__itemName," to value ",self.__itemValue
        else:
            self.__domNode.updateElementValue(str(self.__itemValue))
            # # # print MODULENAME,"UPDATING ELEMENT ",self.__itemName," to value ",self.__itemValue," __domNode.getName()=",self.__domNode.getName()
            # # # print MODULENAME,"VALUE CHECK ",self.__domNode.getText()
        
        if self.type() is None:
            self.setType(value)
        
    def name(self):
        return self.__itemName
    
    def value(self):
        return self.__itemValue
        
    # Sets type of the values. The type can be set only when the XML simulation
    # is loaded and cannot be changed!

    def setType(self, value):
        # Doesn't hurt to create a new str object even if self.__itemValue
        # is of type str
        if value != "":
            # str = QString(value)
            str_obj = str(value)


            # Try to convert to Int
            try:
                val = int(str_obj)
                self.__type = "int"
                return
            except ValueError:
                pass

            # val, ok = str_obj.toInt()
            # if ok:
            #     self.__type = "int"
            #     # print "%s: int" % value
            #     return

            # Try to convert to Float
            try:
                val = float(str_obj)
                self.__type = "float"
                return
            except ValueError:
                pass

            # val, ok = str_obj.toFloat()
            # if ok:
            #     self.__type = "float"
            #     # print "%s: float" % value
            #     return

            self.__type = "string"
            # print "%s: string" % value

    # def setType(self, value):
    #     # Doesn't hurt to create a new QString object even if self.__itemValue
    #     # is of type QString
    #     if value != "":
    #         # str = QString(value)
    #         str = QString(value)
    #
    #         # Try to convert to Int
    #         val, ok = str.toInt()
    #         if ok:
    #             self.__type = "int"
    #             #print "%s: int" % value
    #             return
    #
    #         # Try to convert to Float
    #         val, ok = str.toFloat()
    #         if ok:
    #             self.__type = "float"
    #             #print "%s: float" % value
    #             return
    #
    #         self.__type = "string"
    #         #print "%s: string" % value
    
    def type(self):
        return self.__type
                
    def dumpDomNode(self, s, node):
        s = str(s) + "\n" 
        s += "Element: \n"
        
        if node.isText():
            name = node.parentNode().nodeName()
        else:
            name = node.nodeName()

        s += "  %s: %s\n" % (name, node.nodeValue())
        if node.hasAttributes():
            s += "Attribute(s): %s\n" % node.attributes().count()
            for i in range(node.attributes().count()):
                s += "    %s: %s\n" % (str(node.attributes().item(i).nodeName()), str(node.attributes().item(i).nodeValue()))
            
        print s


# # Very important!
# # Populates tree with TreeItem based on the 
# # QDomDocument and returns tree root item

# def treeNode(itemNode,_superParent=None):
    # import XMLUtils
    # # itemNode can only be Element!
    # # itemNode is of type CC3DXMLElement
    # if not itemNode:
        # return None
        
    # node = itemNode
    # print "creating TreeItem ",node.name," cdata=",node.cdata
    
    # tNode=TreeItem(node.name, node.cdata)
    
    # tNode.setCC3DXMLElement(node) # domNode holds reference to current CC3DXMLElement
    
    # childIdx = 0
    # # setting superParents -they are either Potts, Metadata, Plugin or Steppable. SuperParents are used in steering. Essentially when we change any TreeItem in the TreeView
    # # we can quickly extract name of the superParent and tell CC3D to reinitialize module that superParent describes. Otherwise if we dont keep track of super parents we would either have to do:
    # # 1. time consuming and messy tracking back of which module needs to be changed in response to change in one of the parameter
    # # or
    # # 2. reinitialize all modules each time a single parameter is changed
    
    # superParent=_superParent
    # if not _superParent:
        # if node.name in ("Plugin","Steppable","Potts","Metadata"):
            # superParent=tNode
            

    
    # tNode.setSuperParent(superParent)
    
    
    
    # # setting value for the Potts element
    # if node.name=="Potts":
        # tNode.setValue("Potts")
    
    # if node.name=="Metadata":
        # tNode.setValue("Metadata")    
    
    # # handling opening elements for Plugin and Steppables
    # if node.name=="Plugin":
        # tNode.setValue(node.getAttribute("Name"))
    # if node.name=="Steppable":
        # tNode.setValue(node.getAttribute("Type"))    
    


    
    # if node.attributes.size()>0:
        
        # for attributeName in node.attributes:
            # # print " value x=",node.attributes['x']    
            # # print " value y=",node.attributes['y']    
            # # print " value z=",node.attributes['z']    
            # if node.name=="Plugin" and attributeName=="Name":
                # continue
            # if node.name=="Steppable" and attributeName=="Type":
                # continue
            # # print "attributeName ",attributeName, " value=",node.attributes[attributeName]    
            # treeChild = TreeItem(attributeName, node.attributes[attributeName]) #attribute name, attribute value pair
            # treeChild.setCC3DXMLElement(node)
            # treeChild.setSuperParent(superParent)
            # treeChild.setElementType("attribute")
            
            # tNode.addChild(treeChild)
            
            # childIdx+=1
        

    
    # children=XMLUtils.CC3DXMLListPy(node.children)
    # for child in children:
        # print "element=",child
        # tChild=treeNode(child,superParent)
        # tNode.addChild(tChild)
        
        # print "tChild.domNode().getName()=",tChild.domNode().getName()," parentName=",tChild.parent().name()
        
        # childIdx+=1
        
    # return tNode    
# # Very important!
# # Populates tree with TreeItem based on the 
# # QDomDocument and returns tree root item
    
def treeNode(itemNode,_superParent=None):
    import XMLUtils
    import CC3DXML
    # itemNode can only be Element!
    # itemNode is of type CC3DXMLElement
    if not itemNode:
        return None
        
    node = itemNode
#    print MODULENAME," treeNode(): creating TreeItem ",node.name," cdata=",node.cdata
#    if node.name == 'Plugin':
#      print MODULENAME,'     rwh: exception here', foo
    
    tNode=TreeItem(node.name, node.cdata)

    # Setting item values for Potts, Metadata,  Plugins and Steppables. Those settings are for display purposes only and do not affect CC3DXML element that Compucell3d uses to configure the data
    # all we do is to label Potts tree element as Posst and Plugin and Steppables are named using their names extracted from xml file.
    # we are using _itemValue to label label those elements and do not change cdata of the CC3DXMLelement
    
    if node.name=="Potts":
        tNode.setItemValueOnly("Potts")

    if node.name=="Metadata":
        tNode.setItemValueOnly("Metadata")
        
    # handling opening elements for Plugin and Steppables
    if node.name=="Plugin":
        tNode.setItemValueOnly(node.getAttribute("Name"))
    if node.name=="Steppable":
        tNode.setItemValueOnly(node.getAttribute("Type"))    
    
    tNode.setCC3DXMLElement(node) # domNode holds reference to current CC3DXMLElement
    
 
    # setting superParents -they are either Potts, Plugin or Steppable. SuperParents are used in steering. Essentially when we change any TreeItem in the TreeView
    # we can quickly extract name of the superParent and tell CC3D to reinitialize module that superParent describes. Otherwise if we dont keep track of super parents we would either have to do:
    # 1. time consuming and messy tracking back of which module needs to be changed in response to change in one of the parameter
    # or
    # 2. reinitialize all modules each time a single parameter is changed
    
    superParent=_superParent
    if not _superParent:
        if node.name in ("Plugin","Steppable","Potts", "Metadata"):
            superParent=tNode
    
    tNode.setSuperParent(superParent)
     
    
    
    
    # FOR AN UNKNOWN REASON "regular" map iteration does not work so i had to implement by hand iterators in C++ and Python to walk thrugh all the elementnts in the map<string,string> in python
    
    attribList=XMLUtils.MapStrStrPy(node.attributes)
    # # # print "attributes=",attribList
    for attr in attribList:
        attributeName=attr[0]
        attributeValue=attr[1]
        if node.name=="Plugin" and attributeName=="Name":
            continue
        if node.name=="Steppable" and attributeName=="Type":
            continue
        # print "attributeName ",attributeName, " value=",node.attributes[attributeName]    
        treeChild = TreeItem(attributeName, attributeValue) #attribute name, attribute value pair
        treeChild.setCC3DXMLElement(node)
        treeChild.setSuperParent(superParent)
        treeChild.setElementType("attribute")
        # if superParent is not None:
            # print MODULENAME,"treeChild.domNode().getName()=",treeChild.domNode().getName()
            # ," parentName=",treeChild.parent().name()," super parent=",treeChild.getSuperParent().name()
        tNode.addChild(treeChild)
    
    
    children=XMLUtils.CC3DXMLListPy(node.children)
    for child in children:
        # # # print "element=",child
        tChild=treeNode(child,superParent)
        tNode.addChild(tChild)
        
        # # # if tChild.getSuperParent() is not None:
            # # # print MODULENAME,"tChild.domNode().getName()=",tChild.domNode().getName()," parentName=",tChild.parent().name()," super parent=",tChild.getSuperParent().name()
        
    return tNode    

      
