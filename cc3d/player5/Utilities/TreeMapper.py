from cc3d.core import XMLUtils
from cc3d.core.XMLUtils import dictionaryToMapStrStr as d2mss

MODULENAME = 'TreeMapper.py:--'


class TreeItem:
    def __init__(self, name, value=""):
        self.__itemName = name
        self.__itemValue = value  # The type of self.__itemValue is string

        # Holds the type of the value: "int", "float", "string" 
        self.__type = None
        # Holds the type of element: "attribute", "element"
        self.__elementType = "element"

        self.__childItems = []
        self.__parentItem = None  # By default parent item is None (as root item)!

        # setting superParents -they are either Potts, Metadata, Plugin or Steppable.
        # SuperParents are used in steering. Essentially when we change any TreeItem in the TreeView
        # we can quickly extract name of the superParent and tell CC3D to reinitialize module that superParent
        # describes. Otherwise if we dont keep track of super parents we would either have to do:
        # 1. time consuming and messy tracking back of which module needs to be changed in response
        # to change in one of the parameter
        # or
        # 2. reinitialize all modules each time a single parameter is changed

        self.__superParent = None

        # Very important! Holds the reference to the CC3DXMLElement that correspond to this TreeItem
        self.__domNode = None

        # this is essentially a C++ pointer to CC3DXMLElement converted to long.
        # It is used to identify TreeItems. Otherwise differentiating between different treeItems would be less elegant
        self.__treeItemId = None

        self.setType(value)

    def getTreeItemId(self):
        return self.__treeItemId

    def setCC3DXMLElement(self, _cc3dXMLElement):
        self.__domNode = _cc3dXMLElement
        self.__treeItemId = _cc3dXMLElement.getPointerAsLong()

    def setElementType(self, _elementType):
        self.__elementType = _elementType

    def setSuperParent(self, _superParent):
        self.__superParent = _superParent

    def getSuperParent(self):
        return self.__superParent

    def parent(self):
        return self.__parentItem

    def addChild(self, child):
        if child is not None:
            # Parent is set when the child is added!
            child.setParent(self)
            self.__childItems.append(child)
            # self.__childItems[i] = child

    def removeChild(self, childName):
        for i in self.__childItems:
            if self.__childItems[i].name() == childName:
                del self.__childItems[i]

    def child(self, i):
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

    def setItemValueOnly(self, value):
        self.__itemValue = value

    def setValue(self, value):

        self.__itemValue = value
        if self.__elementType == "attribute":
            # todo 5 - check if this ever gets called - most likely during steering from the player
            self.__domNode.updateElementAttributes(d2mss({self.__itemName: str(self.__itemValue)}))
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

            # Try to convert to Float
            try:
                val = float(str_obj)
                self.__type = "float"
                return
            except ValueError:
                pass

            self.__type = "string"

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
                s += "    %s: %s\n" % (
                str(node.attributes().item(i).nodeName()), str(node.attributes().item(i).nodeValue()))

        print(s)



def treeNode(itemNode, _superParent=None):
    # itemNode can only be Element!
    # itemNode is of type CC3DXMLElement
    if not itemNode:
        return None


    try:
        itemNode.name
        node = itemNode
    except AttributeError:
        node = itemNode.root

    # node = itemNode

    t_node = TreeItem(node.name, node.cdata)


    # Setting item values for Potts, Metadata,  Plugins and Steppables.
    # Those settings are for display purposes only and do not affect CC3DXML element that
    # Compucell3d uses to configure the data
    # all we do is to label Potts tree element as Posst and Plugin and Steppables are named
    # using their names extracted from xml file.
    # we are using _itemValue to label label those elements and do not change cdata of the CC3DXMLelement

    if node.name == "Potts":
        t_node.setItemValueOnly("Potts")

    if node.name == "Metadata":
        t_node.setItemValueOnly("Metadata")

    # handling opening elements for Plugin and Steppables
    if node.name == "Plugin":
        t_node.setItemValueOnly(node.getAttribute("Name"))
    if node.name == "Steppable":
        t_node.setItemValueOnly(node.getAttribute("Type"))

    t_node.setCC3DXMLElement(node)  # domNode holds reference to current CC3DXMLElement

    # setting superParents -they are either Potts, Plugin or Steppable. SuperParents are used in steering.
    # Essentially when we change any TreeItem in the TreeView
    # we can quickly extract name of the superParent and tell CC3D to reinitialize module that superParent describes.
    # Otherwise if we dont keep track of super parents we would either have to do:
    # 1. time consuming and messy tracking back of which module needs to be changed in response
    # to change in one of the parameter
    # or
    # 2. reinitialize all modules each time a single parameter is changed

    superParent = _superParent
    if not _superParent:
        if node.name in ("Plugin", "Steppable", "Potts", "Metadata"):
            superParent = t_node

    t_node.setSuperParent(superParent)

    # FOR AN UNKNOWN REASON "regular" map iteration does not work so i had to implement by hand iterators
    # in C++ and Python to walk through all the elements in the map<string,string> in python

    if node.attributes.size():
        for attr_combo in node.attributes.items():
            attribute_name = attr_combo[0]
            attribute_value = attr_combo[1]
            if node.name == "Plugin" and attribute_name == "Name":
                continue
            if node.name == "Steppable" and attribute_name == "Type":
                continue

            tree_child = TreeItem(attribute_name, attribute_value)  # attribute name, attribute value pair
            tree_child.setCC3DXMLElement(node)
            tree_child.setSuperParent(superParent)
            tree_child.setElementType("attribute")
            t_node.addChild(tree_child)

    children = XMLUtils.CC3DXMLListPy(node.children)
    for child in children:

        t_child = treeNode(child, superParent)
        t_node.addChild(t_child)

    return t_node
