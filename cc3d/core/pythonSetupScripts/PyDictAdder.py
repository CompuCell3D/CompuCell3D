import copy
class DictAdder:
    def __init__(self):
            self.dict_template={}
    def addAttribute(self):
        temp_copy=copy.deepcopy(self.dict_template)
        return temp_copy

#import sys
#def destroyAttributeFcn(_pyAttrib):
    #print "ref Count from Python=",sys.getrefcount(_pyAttrib)
    ##del _pyAttrib
    #print " After del ref Count from Python=",sys.getrefcount(_pyAttrib)
    

