from PyQt5.QtGui import *
from PyQt5.QtCore import *
from cc3d.player5.Utilities.TreeMapper import *

PROPERTY, VALUE = list(range(2))


# The SimModel is specific to the CompuCell3D XML simulation models!

class SimModel(QAbstractItemModel):

    def __init__(self, domDoc, parent=None):

        QAbstractItemModel.__init__(self, parent)

        # Store the QDomDocument (root of the document tree) in the private attribute
        self.__domDoc = domDoc
        self.__printFlag = False

        self.__rootItem = treeNode(domDoc)

        if self.__rootItem:
            print("THIS IS ROOT ITEM=", self.__rootItem.name())
            print("ROOT ITEM DOMNode=", self.__rootItem.domNode().getName())
            self.checkSanity()
            # text = raw_input('Enter text here->')

        self.__isDirty = False
        self.__dirtyModules = {}
        self.__headers = ["Property", "Value"]

    def setPrintFlag(self, _flag):
        self.__printFlag = _flag

    def addDirtyModule(self, _moduleCategory, name):
        if _moduleCategory in self.__dirtyModules:
            self.__dirtyModules[_moduleCategory][name] = 0
        else:
            self.__dirtyModules[_moduleCategory] = {name: 0}

    def getDirtyModules(self):
        return self.__dirtyModules

    def columnCount(self, parent):  # interface: done
        # if parent.isValid():
        # return parent.internalPointer().columnCount()
        # else:
        # return self.__rootItem.columnCount()
        return len(self.__headers)

    def treeItemFromIndex(self, _itemIndex):
        if _itemIndex.isValid():
            return _itemIndex.internalPointer()
        else:
            return self.__rootItem

    def data(self, index, role=Qt.DisplayRole):  # interface: done
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

    def headerData(self, section, orientation, role):  # interface: done
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if 0 <= section <= len(self.__headers):
                return QVariant(self.__headers[section])

        return QVariant()

    def index(self, row, column, parentIndex):  # interface: done
        if self.__rootItem is None:
            return QModelIndex()
        parentTreeItem = self.treeItemFromIndex(parentIndex)
        return self.createIndex(row, column, parentTreeItem.child(row))

    def parent(self, childIndex):  # interface: done
        if not childIndex.isValid():
            return QModelIndex()

        childItem = childIndex.internalPointer()
        parentItem = childItem.parent()

        if parentItem is None:
            return QModelIndex()

        grandparentItem = parentItem.parent()
        if grandparentItem is None:
            return QModelIndex()

        index = self.createIndex(parentItem.siblingIdx(), 0, parentItem)

        return index

    def rowCount(self, parentIndex):  # interface: done
        if self.__rootItem is None:
            return 0
        elif not parentIndex.isValid():
            parentItem = self.__rootItem
        else:
            parentItem = parentIndex.internalPointer()

        return parentItem.childCount()

    # Sets the data of the tree node. 
    # (Fix it?) Even if the data has not been changed it still sets self.__isDirty = True

    def conv_to_int(self, obj):
        try:
            int_obj = int(obj)
            return int_obj, True
        except ValueError:
            return None, False

    def conv_to_float(self, obj):
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
                    print("dirty modules=", self.__dirtyModules)
                    print("item SuperParent type=", item.getSuperParent().name(), " type=",
                          item.getSuperParent().value())
                    print("item.domNode()=", item.domNode())
                    print("item.domNode().getName()=", item.domNode().getName())
                    print("item.domNode().getText()=", item.domNode().getText())
                    # item.domNode().setNodeValue(str) #setNodeValue()

            self.__isDirty = True  # Variable that chacks if data has been modified!
            self.dataChanged.emit(index, index)
            # self.emit(SIGNAL("dataChanged(QModelIndex,QModelIndex)"), index, index)
            # self.emit(SIGNAL("dataChanged(QModelIndex,QModelIndex)"), index, index)
            return True

        return False

    def domDocIsDirty(self):
        return self.__isDirty

    def setDirty(self, dirty):
        self.__isDirty = dirty

    def domDoc(self):
        return self.__domDoc

    def checkSanity(self):
        pass


# class XMLElemAdapter:
#     def __init__(self, reference_elem=None, super_parent=None):
#
#         # reference to the C++ XML element representation
#
#         self.__allowed_assignment_properties = []
#         self.__reference_elem = reference_elem
#         # usually stores reference to Plugin, Steppable or Potts elements - modules that will have to be initialized
#         self.__super_parent = super_parent
#
#         self._dirty_flag = False
#         if reference_elem is not None:
#             self.init_attributes()
#
#
#
#     def init_attributes(self):
#         """
#         adds new properties to the object base on attributes of the self.__reference_element
#         The goal is to give user an object that is easy to manipulate
#
#         :return: None
#         """
#         if self.__reference_elem is None:
#             return
#
#         attributes = self.__reference_elem.getAttributes()
#
#         setattr(self, 'cdata', self.__reference_elem.cdata)
#         self.__allowed_assignment_properties.append('cdata')
#
#         for attr_key in attributes.keys():
#             setattr(self, attr_key, attributes[attr_key])
#             self.__allowed_assignment_properties.append(attr_key)
#
#         # self.__setattr__ = self.__attr_setter
#         self.attribs_initialized = True
#
#     def set_dirty(self, flag=True):
#         self._dirty_flag = flag
#
#
#     @property
#     def dirty(self):
#         return self._dirty_flag
#
#     # def __attr_setter(self, key, value):
#     def __setattr__(self, key, value):
#
#         try:
#             self.attribs_initialized
#             be_selective = True
#         except (KeyError,AttributeError):
#             be_selective = False
#             print('self.attribs_initialized does not exist')
#
#         if not be_selective:
#             self.__dict__[key] = value
#         else:
#             if key == '_dirty_flag':
#                 self.__dict__['_dirty_flag'] = value
#                 return
#             if key in self.__allowed_assignment_properties:
#                 self.__dict__[key] = value
#                 self.__dict__['_dirty_flag'] = True
#                 # print
#             else:
#                 raise AttributeError('Attribute {attr} is not assignable. '
#                                      'The list of assignable attributes is: {attr_list}'.format(
#                     attr=key,
#                     attr_list=self.__allowed_assignment_properties
#
#                 ))
#
#         # # setattr(self,key,value)
#         # return
#         #
#         # try:
#         #     self.__allowed_assignment_properties
#         # except AttributeError:
#         #     return
#         #
#         # if key in self.__allowed_assignment_properties:
#         #     self.__dict__[key] = value
#         # else:
#         #     raise AttributeError('Attribute {attr} is not assignable. '
#         #                          'The list of assignable attributes is: {attr_list}'.format(
#         #         attr=key,
#         #         attr_list=self.__allowed_assignment_properties
#         #
#         #     ))
#         #
#         # # if self.__dict__.get("_locked", False) and name == "x":
#         # #     raise AttributeError("MyClass does not allow assignment to .x member")
#         # # self.__dict__[name] = value
#
#
# class IdLocator:
#     def __init__(self, root_elem):
#
#         self.root_elem = root_elem
#         self.node_stack = []
#         self.super_parent_stack = []
#
#         # dict labeled by the value of the id element
#         self.id_elements_dict = {}
#
#         self.recently_accessed_elems = {}
#         # self.recently_modified_elems = {}
#
#     def locate_id_elements(self):
#         """
#
#         :return:
#         """
#         self.walk_and_locate_id_elements(elem=self.root_elem)
#
#     def walk_and_locate_id_elements(self, elem):
#         """
#
#         :return:
#         """
#
#         id_elem_list = []
#         children = elem.children
#         self.node_stack.append(elem)
#
#         print('elem=', elem.name)
#
#         if elem.name in ['Potts', 'Plugin', 'Steppable']:
#             self.super_parent_stack.append(elem)
#             print('adding super_parent=', elem.name)
#
#         # direct iteration like for child_elem in  root_elem.children: does not work with SWIG
#         for child_idx in range(children.size()):
#             child = children[child_idx]
#             attributes = child.getAttributes()
#             if 'id' in attributes.keys():
#                 elem_with_id = XMLElemAdapter(reference_elem=child, super_parent=self.super_parent_stack[-1])
#                 id_attr = child.getAttribute('id')
#                 self.id_elements_dict[id_attr] = elem_with_id
#
#             # print(attributes)
#             self.walk_and_locate_id_elements(elem=child)
#
#         popped_elem = self.node_stack.pop()
#         if popped_elem.name in ['Potts', 'Plugin', 'Steppable']:
#             self.super_parent_stack.pop()
#
#     def get_xml_elem(self, id):
#         try:
#             elem = self.id_elements_dict[id]
#             self.recently_accessed_elems[id] = elem
#             return elem
#         except KeyError:
#             return None
#
#
# if __name__ == '__main__':
#
#     node_stack = []
#
#
#     def locate_id_elements(root_elem, node_stack):
#         id_elem_list = []
#         children = root_elem.children
#         node_stack.append(root_elem)
#         print('elem=', root_elem.name)
#
#         # direct iteration like for child_elem in  root_elem.children: does not work with SWIG
#         for child_idx in range(children.size()):
#             child = children[child_idx]
#             attributes = child.getAttributes()
#             # print(attributes)
#             locate_id_elements(root_elem=child, node_stack=node_stack)
#
#         node_stack.pop()
#
#         # for child_elem in  root_elem.children:
#         #     # print(child_elem)
#         #     break
#         print
#
#
#     from cc3d import CompuCellSetup
#
#     file_name = r'd:\CC3D_PY3_GIT\CompuCell3D\core\DemosNew\ExtraFields\Simulation\ExtraFields.xml'
#     cc3d_xml_2_obj_converter = CompuCellSetup.parseXML(file_name)
#
#     # locate_id_elements(cc3d_xml_2_obj_converter.root, node_stack=node_stack)
#     # print('node_stack=',node_stack)
#
#     id_locator = IdLocator(root_elem=cc3d_xml_2_obj_converter.root)
#     id_locator.locate_id_elements()
#
#     vol_cond_elem = id_locator.get_xml_elem('vol_cond')
#     vol_cond_elem.LambdaVolume=3.0
#
#     diff_const_elem = id_locator.get_xml_elem('fgf_diff_constant')
#
#
#     print
#
#     #
#     # sim_model = SimModel(domDoc=cc3d_xml_2_obj_converter)
#     #
#     # print
