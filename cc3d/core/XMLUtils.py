from xml.parsers import expat
from cc3d.cpp import CC3DXML
from cc3d.cpp.CC3DXML import *


def dictionaryToMapStrStr(_dictionary:dict)->object:
    """
    converts python dictionaty to c++ map (std::map<std:string, std:string>)
    :param _dictionary:
    :return:
    """
    mapStrStr = CC3DXML.MapStrStr()
    for key in _dictionary.keys():
        # mapStrStr[key.encode()]=str(_dictionary[key])
        mapStrStr[key] = str(_dictionary[key])
    return mapStrStr


class ElementCC3D:
    def __init__(self, _name, _attributes={}, _cdata=""):
        self.CC3DXMLElement = CC3DXML.CC3DXMLElement(str(_name), dictionaryToMapStrStr(_attributes), str(_cdata))
        self.childrenList = []

    def ElementCC3D(self, _name, _attributes={}, _cdata=""):
        self.childrenList.append(ElementCC3D(_name, _attributes, _cdata))
        self.CC3DXMLElement.addChild(self.childrenList[-1].CC3DXMLElement)
        return self.childrenList[-1]

    def addComment(self, _comment):
        self.CC3DXMLElement.addComment(_comment)

    def commentOutElement(self):
        self.CC3DXMLElement.commentOutElement()

    def getCC3DXMLElementString(self):
        return self.CC3DXMLElement.getCC3DXMLElementString()

    def add_child(self, _child: ElementCC3D) -> None:
        self.childrenList.append(_child)
        self.CC3DXMLElement.addChild(self.childrenList[-1].CC3DXMLElement)


class CC3DXMLListPy:
    def __init__(self, _list):
        self.elementList = _list

    def __iter__(self):
        return CC3DXMLListIteratorPy(self)

    def getBaseClass(self):
        return self.elementList.getBaseClass()


class CC3DXMLListIteratorPy:
    def __init__(self, _cc3dXMLElementListPy):
        self.elementList = _cc3dXMLElementListPy.elementList

        self.invItr = CC3DXML.CC3DXMLElementListIterator()

        self.invItr.initialize(self.elementList)
        self.invItr.setToBegin()

    def __next__(self):
        if not self.invItr.isEnd():
            self.element = self.invItr.getCurrentRef()
            self.invItr.next()
            return self.element
        else:
            raise StopIteration

    def __iter__(self):
        return self


# FOR AN UNKNOWN REASON "regular" map iteration does not work so we had to
# implement by hand iterators in C++ and Python to walk thrugh all the elementnts in the map<string,string> in python
class MapStrStrPy:
    def __init__(self, _map):
        self.map = _map

    def __iter__(self):
        return MapStrStrIteratorPy(self)

    def getBaseClass(self):
        return self.elementList.getBaseClass()


class MapStrStrIteratorPy:
    def __init__(self, _mapStrStrElementPy):
        self.map = _mapStrStrElementPy.map
        #         self.invItr=CC3DXML.STLPyIteratorCC3DXMLElementList()
        self.invItr = CC3DXML.MapStrStrIterator()

        self.invItr.initialize(self.map)
        self.invItr.setToBegin()

    def next(self):
        if not self.invItr.isEnd():
            self.element = self.invItr.getCurrentRef()
            self.invItr.next()
            return self.element
        else:
            raise StopIteration

    def __iter__(self):
        return self


class XMLAttributeList:
    def __init__(self, _xmlElement):
        self.xmlElement = _xmlElement

    def __iter__(self):
        return XMLAttributeIterator(self)


class XMLAttributeIterator:
    def __init__(self, _xmlAttributeList):
        self.xmlElement = _xmlAttributeList.xmlElement

        # have to store attributes in a local variable so that it does not get garbage collected
        self.attributes = self.xmlElement.getAttributes()
        self.itr = CC3DXML.MapStrStrIterator()

        self.itr.initialize(self.attributes)
        self.itr.setToBegin()

    def next(self):
        if not self.itr.isEnd():
            self.attributeNameValuePair = self.itr.getCurrentRef()
            self.itr.next()
            return self.attributeNameValuePair
        else:
            raise StopIteration

    def __iter__(self):
        return self


class Xml2Obj(object):
    ''' XML to Object converter '''

    def __init__(self):
        self.root = None
        self.nodeStack = []
        self.elementInventory = []

    def StartElement(self, name, attributes):
        'Expat start element event handler'
        element = CC3DXML.CC3DXMLElement(str(name), dictionaryToMapStrStr(attributes))

        # Push element onto the stack and make it a child of parent
        if self.nodeStack:
            parent = self.nodeStack[-1]
            parent.addChild(element)
        else:
            self.root = element
        self.nodeStack.append(element)
        self.elementInventory.append(element)


    def EndElement(self, name):

        self.nodeStack.pop()

    def CharacterData(self, data):

        if data.strip():
            # data = data.encode()
            data = str(data)
            element = self.nodeStack[-1]
            element.cdata += data

    def Parse(self, filename):
        # Create an Expat parser
        Parser = expat.ParserCreate()

        # Set the Expat event handlers to our methods
        Parser.StartElementHandler = self.StartElement
        Parser.EndElementHandler = self.EndElement
        Parser.CharacterDataHandler = self.CharacterData
        file_handle = open(filename)

        ParserStatus = Parser.Parse(file_handle.read(), 1)
        file_handle.close()

        return self.root

    def ParseString(self, _string):
        # Create an Expat parser
        Parser = expat.ParserCreate()
        # Set the Expat event handlers to our methods
        Parser.StartElementHandler = self.StartElement
        Parser.EndElementHandler = self.EndElement
        Parser.CharacterDataHandler = self.CharacterData
        # Parse the XML File
        ParserStatus = Parser.Parse(_string, 1)

        return self.root
