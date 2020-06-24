class DOMBase(object):

    def __init__(self, _name=''):

        # this dictionary lists attributes expected in the xml, their type and default value        

        # format {attrName:(type,defaultValue)} type is Python type name

        self.attrNameToTypeDict = {}

        self.__name = _name

    def fromDOMElem(self, _domElem):

        for attrName, attrTypeDefTypeTuple in self.attrNameToTypeDict.items():

            if _domElem.hasAttribute(attrName):

                try:

                    setattr(self, attrName, attrTypeDefTypeTuple[0](_domElem.getAttribute(attrName)))

                except:

                    # print 'Could not convert ',attrName,' to ',attrTypeDefTypeTuple[0],' using default value of ',attrTypeDefTypeTuple[1]

                    setattr(self, attrName, attrTypeDefTypeTuple[1])

            else:

                setattr(self, attrName, attrTypeDefTypeTuple[1])

    def __str__(self):

        s = self.__name + '\n'

        for attrName, attrTypeDefTypeTuple in self.attrNameToTypeDict.items():
            s += attrName + ' = ' + str(getattr(self, attrName)) + '\n'

        return s
