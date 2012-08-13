import sys
sys.path.append("d:\Program Files\COMPUCELL3D_3.3.1\lib\python")

from xml.parsers import expat
class Element:
    ''' A parsed XML element '''
    def __init__(self, name, attributes):
        # Record tagname and attributes dictionary
        self.name = name
        self.attributes = attributes
        # Initialize the element's cdata and children to empty
        self.cdata = ''
        self.children = []
    def addChild(self, element):
        self.children.append(element)
    def getAttribute(self, key):
        return self.attributes.get(key)
    def getData(self):
        return self.cdata
    def getElements(self, name=''):
        if name:  
            return [c for c in self.children if c.name == name]
        else:
            return list(self.children)
class Xml2Obj(object):
    ''' XML to Object converter '''
    def __init__(self):
        self.root = None
        self.nodeStack = []
    def StartElement(self, name, attributes):
        'Expat start element event handler'
        # Instantiate an Element object
        element = Element(name.encode(), attributes)
        print "start_element:", name, attributes
        # Push element onto the stack and make it a child of parent
        if self.nodeStack:
            parent = self.nodeStack[-1]
            parent.addChild(element)
        else:
            self.root = element
        self.nodeStack.append(element)
        print "self.nodeStack=",self.nodeStack[-1].name
    def EndElement(self, name):
        'Expat end element event handler'
        print "End Element", name

        self.nodeStack.pop()
    def CharacterData(self, data):
        'Expat character data event handler'

        if data.strip():
            data = data.encode()
            print "Character Data",repr(data)
            element = self.nodeStack[-1]
            element.cdata += data
    def Parse(self, filename):
        # Create an Expat parser
        Parser = expat.ParserCreate()
        # Set the Expat event handlers to our methods
        Parser.StartElementHandler = self.StartElement
        Parser.EndElementHandler = self.EndElement
        Parser.CharacterDataHandler = self.CharacterData
        # Parse the XML File
        ParserStatus = Parser.Parse(open(filename).read(),1)
        return self.root

class Xml2ObjCpp(object):
    ''' XML to Object converter '''
    def __init__(self):
        self.root = None
        self.nodeStack = []
        self.elementInventory=[]
    def StartElement(self, name, attributes):
        'Expat start element event handler'
        # Instantiate an Element object
        import CC3DXML
#         element = Element(name.encode(), attributes)
        element = CC3DXML.CC3DXMLElement(name.encode(), self.dictionaryToMapStrStr(attributes))
        print "start_element:", name, attributes
        # Push element onto the stack and make it a child of parent
        if self.nodeStack:
            parent = self.nodeStack[-1]
            parent.addChild(element)
        else:
            self.root = element
        self.nodeStack.append(element)
        self.elementInventory.append(element)
        print "self.nodeStack=",self.nodeStack[-1].name

    def dictionaryToMapStrStr(self,_dictionary):
        import CC3DXML
        mapStrStr=CC3DXML.MapStrStr()
        for key in _dictionary.keys():
#           print "_dictionary[key.encode()]=",_dictionary[key.encode()]," type=",type(key.encode())
            mapStrStr[key.encode()]=_dictionary[key].encode()
        return mapStrStr

    def EndElement(self, name):
        'Expat end element event handler'
        print "End Element", name

        self.nodeStack.pop()
    def CharacterData(self, data):
        'Expat character data event handler'

        if data.strip():
            data = data.encode()
            print "Character Data",repr(data)
            element = self.nodeStack[-1]
            element.cdata += data
    def Parse(self, filename):
        # Create an Expat parser
        Parser = expat.ParserCreate()
        # Set the Expat event handlers to our methods
        Parser.StartElementHandler = self.StartElement
        Parser.EndElementHandler = self.EndElement
        Parser.CharacterDataHandler = self.CharacterData
        # Parse the XML File
        ParserStatus = Parser.Parse(open(filename).read(),1)
        return self.root

class ElementCC3D:
    def __init__(self,_name="",_attributes={},_cdata=""):
        if _name=="":
            self.CC3DXMLElement=None
        else:
            self.CC3DXMLElement=CC3DXML.CC3DXMLElement(str(_name),self.dictionaryToMapStrStr(_attributes),str(_cdata))
        self.childrenList=[]

            
            
    @classmethod
    def wrapElementCC3D(cls,_CC3DXMLElement):
        obj=cls()
        obj.CC3DXMLElement=_CC3DXMLElement
        return obj
    
    def dictionaryToMapStrStr(self,_dictionary):
        import CC3DXML
        mapStrStr=CC3DXML.MapStrStr()
        for key in _dictionary.keys():
            mapStrStr[key.encode()]=str(_dictionary[key])
        return mapStrStr

    def ElementCC3D(self,_name,_attributes={},_cdata=""):
        self.childrenList.append(ElementCC3D(_name,_attributes,_cdata))
        self.CC3DXMLElement.addChild(self.childrenList[-1].CC3DXMLElement)
        return self.childrenList[-1]

    def getFirstElement(self,_name,_attributes={}):
        print "TRYING TO GET FIRST ELEMENT"
        attributesMap=self.dictionaryToMapStrStr(_attributes)
        return self.CC3DXMLElement.getFirstElement(_name,attributesMap)
        # for element in self.childrenList:
            # if element.checkMatch(_name,attributesMap):
                # return element
        # return None
   
class CC3DXMLListPy:
    def __init__(self,_list):
        self.elementList = _list
    def __iter__(self):
        return CC3DXMLListIteratorPy(self)

class CC3DXMLListIteratorPy:
    def __init__(self, _cc3dXMLElementListPy):
        import CC3DXML
        self.elementList = _cc3dXMLElementListPy.elementList
#         self.invItr=CC3DXML.STLPyIteratorCC3DXMLElementList()
        self.invItr=CC3DXML.CC3DXMLElementListIterator()

        self.invItr.initialize(self.elementList)
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




import CC3DXML
import math
a=math.sin(0.5)

def d2mss(_dictionary):
    import CC3DXML
    mapStrStr=CC3DXML.MapStrStr()
    for key in _dictionary.keys():
        mapStrStr[key.encode()]=str(_dictionary[key])
    return mapStrStr


pl1=ElementCC3D("Plugin", {"Type":"Volume"})
pl1.ElementCC3D("LambdaVolume", {},1.0)
pl1.ElementCC3D("TargetVolume", {},25.0)
sc1=pl1.ElementCC3D("NewSection")
sc1.ElementCC3D("NeighborOrder",{},3)
sc1.ElementCC3D("Multiplicity",{},2)



iteratorCpp=CC3DXML.CC3DXMLElementWalker()
iteratorCpp.iterateCC3DXMLElement(pl1.CC3DXMLElement)

print "pl1.CC3DXMLElement.cdata=",pl1.CC3DXMLElement.cdata, " type=",type(pl1.CC3DXMLElement.cdata)

parser = Xml2ObjCpp()
root_element = parser.Parse('cellsort_2D.xml')
print "ROOT ELEMENT=",type(root_element)

pluginElements=root_element.getElements("Plugin")
listPlugin=CC3DXMLListPy(pluginElements)



contact=root_element.getFirstElement("Plugin",d2mss({"Name":"Contact"}))


print "contact=",contact
foundElement=contact.getFirstElement("Energy",d2mss({"Type1":"NonCondensing","Type2":"Condensing"}))
if foundElement:
    print "Found the element,",foundElement
    print "ENERGY VALUE=",foundElement.cdata
    flag=foundElement.checkMatch("Energy",d2mss({"Type1":"NonCondensing","Type2":"Condensing"}))
    print "flag=",flag
    foundElement.updateElementValue(str(110))
    print "ENERGY VALUE=",foundElement.cdata
    foundElement.updateElementAttributes(d2mss({"Type9":"NonCondensing"}))
    flag=foundElement.checkMatch("Energy",d2mss({"Type1":"NonCondensing","Type2":"Condensing"}))
    print "flag=",flag
    
else:
    print "COULD NOT FINF THE ELEMENT"


sys.exit()

# strMap=CC3DXML.MapStrStr()
# strMap["dupa1"]="1"
# 
# 
# dmap = CC3DXML.DoubleMap()
# dmap["hello"] = 1.0
# dmap["hi"] = 2.0



parser = Xml2ObjCpp()
root_element = parser.Parse('cellsort_2D.xml')
print "ROOT ELEMENT=",type(root_element)

# print "\n\n\n\n\nELEMENTS"
# elements=root_element.getElements()
# for element in elements:
#     print "element name=", element.name, " cdata=", element.cdata

def iterate_tree(element):
    childrenList=element.getElements()
    if(childrenList):
        print "ELEMENT:",element.name," HAS CHILDREN"
        for children in childrenList:
            iterate_tree(children)
    else:
        print "element=", element.name," CDATA=",element.cdata, "type=", type(element.cdata)

print " n\n\n\n ITERATION"
# iterate_tree(root_element)




walker=CC3DXML.CC3DXMLElementWalker()
walker.iterateCC3DXMLElement(root_element)

pluginElements=root_element.getElements("Plugin")
listPlugin=CC3DXMLListPy(pluginElements)

for element in listPlugin:
#    d_elem=CC3DXML.derefCC3DXMLElement(element)
    print "element=",element.name
#    ,d_elem.name



sys.exit()
list_string = CC3DXML.getListStringWrapped()

list_string_py=CC3DXML.ListString()
list_string_py.push_back("dupa111111111")
print "FIRST ELEMENT ",CC3DXML.getFirstElement(list_string_py)

# print "FIRST ELEMENT LIST STRING ",CC3DXML.getFirstElement(list_string)

t=CC3DXML.getTry()
print "t.a=",t.a

print "FIRST ELEMENT LIST STRING ",CC3DXML.getFirstElement(t.listString)


print "FIRST ELEMENT LIST STRING ",CC3DXML.getFirstElement(list_string)



