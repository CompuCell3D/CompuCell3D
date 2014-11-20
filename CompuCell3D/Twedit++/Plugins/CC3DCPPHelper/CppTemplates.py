import re
import string

class CppTemplates:
    def __init__(self):
        self.cppTemplatesDict={}
        
        self.initCppTemplates()
        
        
    def getCppTemplatesDict(self):
        return self.cppTemplatesDict

    def generateCACMakeFile(self,_features={}):
        
        if 'Module' in _features.keys():
            text=''
            # if _features['codeLayout']=='developerzone':
                # text=self.cppTemplatesDict["CMakePluginDeveloperZone"]
            # else:
            text=self.cppTemplatesDict["CACMakeModule"]
            moduleName=_features['Module']
            text=re.sub("MODULE_CORE_NAME",moduleName,text) 
            pythonWrap=''
            try:
                if not _features['PythonWrap']:
                    pythonWrap='SWIG OFF'
            except LookupError,e:
                pass
            text=re.sub("PYTHON_WRAP",pythonWrap,text)                 
            # try:
                # extraAttribFlag=_features['ExtraAttribute']
                # text=re.sub("EXTRA_ATTRIBUTE_HEADER",pluginName+'Data.h',text)                     
            # except LookupError,e:
                # text=re.sub("EXTRA_ATTRIBUTE_HEADER",'',text)                     
            return text    
                
        else: # steppable
            return ''
        
    def generateModuleDLLSpecifier(self,_features={}):
    
        replaceLabelList=[]
        moduleName=_features['Module']
                
        moduleDLLSpecifierText=self.cppTemplatesDict["moduleDLLSpecifier"]
        
        MODULE_NAME_CORE=moduleName
        replaceLabelList.append(['MODULE_NAME_CORE',MODULE_NAME_CORE])
        
        MODULE_NAME_CAPITALS=moduleName.upper()
        replaceLabelList.append(['MODULE_NAME_CAPITALS',MODULE_NAME_CAPITALS])
        
        for replaceItems in replaceLabelList:
            moduleDLLSpecifierText=re.sub(replaceItems[0],replaceItems[1], moduleDLLSpecifierText)
            
        return moduleDLLSpecifierText        

    def generateCAModuleHeaderFile(self,_features={}):
    
        replaceLabelList=[]
        moduleName=_features['Module']
        
        moduleHeaderText=self.cppTemplatesDict["CAModuleHeader"]
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        moduleNameVar=lowerFirst(moduleName)
        
        MODULE_NAME_CORE=moduleName
        replaceLabelList.append(['MODULE_NAME_CORE',MODULE_NAME_CORE])
        
        MODULE_NAME_VAR=moduleNameVar
        replaceLabelList.append(['MODULE_NAME_VAR',MODULE_NAME_VAR])
        
        
        
        # set text for replace labels
        IFDEFLABEL=('CA'+moduleName+'module').upper()+'_H'
        replaceLabelList.append(['IFDEFLABEL',IFDEFLABEL])
               
        
        # try:
            # _features['ExtraAttribute']
            # EXTRA_ATTRIB_INCLUDES='#include \"'+moduleName+'Data.h\"'
            # # # # EXTRA_ATTRIB_INCLUDES+="""
# # # # """

            # EXTRA_ATTRIB_ACCESSOR_DEFINE='BasicClassAccessor<'+moduleName+'Data> '+moduleNameVar+'DataAccessor;'
        
            # EXTRA_ATTRIB_ACCESSOR_GET_PTR='BasicClassAccessor<'+moduleName+'Data> * '+'get'+moduleName+'DataAccessorPtr(){return & '+moduleNameVar+'DataAccessor;}'
		


        # except LookupError,e:
            # EXTRA_ATTRIB_INCLUDES=''
            # EXTRA_ATTRIB_ACCESSOR_DEFINE=''
            # EXTRA_ATTRIB_ACCESSOR_GET_PTR=''
            
            
        # replaceLabelList.append(['EXTRA_ATTRIB_INCLUDES',EXTRA_ATTRIB_INCLUDES])
        # replaceLabelList.append(['EXTRA_ATTRIB_ACCESSOR_DEFINE',EXTRA_ATTRIB_ACCESSOR_DEFINE])
        # replaceLabelList.append(['EXTRA_ATTRIB_ACCESSOR_GET_PTR',EXTRA_ATTRIB_ACCESSOR_GET_PTR])        
        
        
        try:
            _features['ProbabilityFunction']
            
            PROBABILITY_FUNCTION_INCLUDE='#include <CA/ProbabilityFunction.h>'            
            PROBABILITY_FUNCTION_INTERFACE='//Probability function interface\n'
            PROBABILITY_FUNCTION_INTERFACE+='        virtual float calculate(const CACell * _sourceCell,const Point3D & _source, const Point3D & _target);'
            PROBABILITY_FUNCTION_BASE='public ProbabilityFunction'
            
        except LookupError,e:
            PROBABILITY_FUNCTION_INCLUDE=''
            PROBABILITY_FUNCTION_INTERFACE=''
            PROBABILITY_FUNCTION_BASE=''
            
            
        replaceLabelList.append(['PROBABILITY_FUNCTION_INCLUDE',PROBABILITY_FUNCTION_INCLUDE])
        replaceLabelList.append(['PROBABILITY_FUNCTION_INTERFACE',PROBABILITY_FUNCTION_INTERFACE])
        replaceLabelList.append(['PROBABILITY_FUNCTION_BASE',PROBABILITY_FUNCTION_BASE])
        
        
        try:
            _features['LatticeMonitor']            
            LATTICE_MONITOR_INCLUDE='#include <CA/CACellStackFieldChangeWatcher.h>'
            LATTICE_MONITOR_INTERFACE='// CACellStackFieldChangeWatcher interface\n'
            LATTICE_MONITOR_INTERFACE+='        virtual void field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack);'
            LATTICE_MONITOR_BASE='public CACellStackFieldChangeWatcher'
        
            
        except LookupError,e:
            LATTICE_MONITOR_INCLUDE=''
            LATTICE_MONITOR_INTERFACE=''
            LATTICE_MONITOR_BASE=''
            
        replaceLabelList.append(['LATTICE_MONITOR_INCLUDE',LATTICE_MONITOR_INCLUDE])            
        replaceLabelList.append(['LATTICE_MONITOR_INTERFACE',LATTICE_MONITOR_INTERFACE])            
        replaceLabelList.append(['LATTICE_MONITOR_BASE',LATTICE_MONITOR_BASE])   
            
        try:
            _features['Steppable']
                        
            STEPPABLE_INCLUDE='#include <CA/CASteppable.h>'
            STEPPABLE_INTERFACE='// Steppable interface\n'        
            STEPPABLE_INTERFACE+='        virtual void start();\n'                    
            STEPPABLE_INTERFACE+='        virtual void step(const unsigned int currentStep);\n'        
            STEPPABLE_INTERFACE+='        virtual void finish();\n'                                
            STEPPABLE_BASE='public CASteppable'
            
            
        except LookupError,e:
            STEPPABLE_INCLUDE=''
            STEPPABLE_INTERFACE=''
            STEPPABLE_BASE=''
            
        replaceLabelList.append(['STEPPABLE_INCLUDE',STEPPABLE_INCLUDE])            
        replaceLabelList.append(['STEPPABLE_INTERFACE',STEPPABLE_INTERFACE])            
        replaceLabelList.append(['STEPPABLE_BASE',STEPPABLE_BASE])                

        DLL_SPECIFIER_INCLUDE='#include \"'+moduleName+'DLLSpecifier.h\"'
        replaceLabelList.append(['DLL_SPECIFIER_INCLUDE',DLL_SPECIFIER_INCLUDE])            
        

        DLL_SPECIFIER_EXPORT=moduleName.upper()+"_EXPORT"
        replaceLabelList.append(['DLL_SPECIFIER_EXPORT',DLL_SPECIFIER_EXPORT])            
        
        # replacing labels with generated text
        for replaceItems in replaceLabelList:
            moduleHeaderText=re.sub(replaceItems[0],replaceItems[1], moduleHeaderText)
            
            
        return moduleHeaderText
        
    def generateCAModuleImplementationFile(self,_features={}):
    
        replaceLabelList=[]
        moduleName=_features['Module']
        
        moduleImplementationText=self.cppTemplatesDict["CAModuleImplementation"]
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        moduleNameVar=lowerFirst(moduleName)
        
        MODULE_NAME_CORE=moduleName
        replaceLabelList.append(['MODULE_NAME_CORE',MODULE_NAME_CORE])
        
        MODULE_NAME_VAR=moduleNameVar
        replaceLabelList.append(['MODULE_NAME_VAR',MODULE_NAME_VAR])
                       
        # try:
            # _features['ExtraAttribute']
            # REGISTER_EXTRA_ATTRIBUTE='potts->getCellFactoryGroupPtr()->registerClass(&'+moduleNameVar+'DataAccessor);'
		
        # except LookupError,e:
            # REGISTER_EXTRA_ATTRIBUTE=''
                        
        # replaceLabelList.append(['REGISTER_EXTRA_ATTRIBUTE',REGISTER_EXTRA_ATTRIBUTE])
        
        try:
            _features['ProbabilityFunction']
            
            
            PROBABILITY_FUNCTION_IMPLEMENTATION="""
float MODULE_NAME_CORE::calculate(const CACell * _sourceCell, const Point3D & _source, const Point3D & _target){
	

	CACellStack * sourceStack = cellFieldS -> get(_source);
	CACellStack * targetStack = cellFieldS -> get(_target);


	float prob = 0.0;
    //IMPLEMENT DETAILS OF PROBABILITY FUNCTION HERE

    return prob;
}               
"""
            PROBABILITY_FUNCTION_IMPLEMENTATION=re.sub('MODULE_NAME_CORE',MODULE_NAME_CORE, PROBABILITY_FUNCTION_IMPLEMENTATION)
        except LookupError,e:
            
            PROBABILITY_FUNCTION_IMPLEMENTATION=''
        
        
        replaceLabelList.append(['PROBABILITY_FUNCTION_IMPLEMENTATION',PROBABILITY_FUNCTION_IMPLEMENTATION])

        try:
            _features['LatticeMonitor']            
            
            LATTICE_MONITOR_IMPLEMENTATION="""            
void MODULE_NAME_CORE::field3DChange(CACell *_movingCell, CACellStack *_sourceCellStack,CACellStack *_targetCellStack){

    //IMPLEMENT LATTICE MONITOR HERE

}
"""            
            LATTICE_MONITOR_IMPLEMENTATION=re.sub('MODULE_NAME_CORE',MODULE_NAME_CORE, LATTICE_MONITOR_IMPLEMENTATION)
        except LookupError,e:            
            LATTICE_MONITOR_IMPLEMENTATION=''
            

        replaceLabelList.append(['LATTICE_MONITOR_IMPLEMENTATION',LATTICE_MONITOR_IMPLEMENTATION])       


        try:
            _features['Steppable']            
            
            STEPPABLE_IMPLEMENTATION="""            
void MODULE_NAME_CORE::start(){

    //IMPLEMENT start FUNCTION HERE

}

void MODULE_NAME_CORE::step(const unsigned int currentStep){

    //IMPLEMENT step FUNCTION HERE

}

void MODULE_NAME_CORE::finish(){

    //IMPLEMENT finish FUNCTION HERE

}

"""            
            STEPPABLE_IMPLEMENTATION=re.sub('MODULE_NAME_CORE',MODULE_NAME_CORE, STEPPABLE_IMPLEMENTATION)
        except LookupError,e:            
            STEPPABLE_IMPLEMENTATION=''
            

        replaceLabelList.append(['STEPPABLE_IMPLEMENTATION',STEPPABLE_IMPLEMENTATION])       
        

        for replaceItems in replaceLabelList:
            moduleImplementationText=re.sub(replaceItems[0],replaceItems[1], moduleImplementationText)
            
        return moduleImplementationText
        
    def generateCAModuleSwigFile(self,_features={}):
        replaceLabelList=[]
        moduleName=_features['Module']
        
        moduleSwigText=self.cppTemplatesDict["CAModuleSwig"]
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        moduleNameVar=lowerFirst(moduleName)
        
        MODULE_NAME_CORE=moduleName
        replaceLabelList.append(['MODULE_NAME_CORE',MODULE_NAME_CORE])
        
        MODULE_NAME_VAR=moduleNameVar
        replaceLabelList.append(['MODULE_NAME_VAR',MODULE_NAME_VAR])
        
        DLL_SPECIFIER_EXPORT=moduleName.upper()+"_EXPORT"
        
        replaceLabelList.append(['DLL_SPECIFIER_EXPORT',DLL_SPECIFIER_EXPORT])

        try:
            _features['ProbabilityFunction']                        
            PROBABILITY_FUNCTION_SWIG_1="""#include <CA/CAProbabilityFunction.h>"""                        
            PROBABILITY_FUNCTION_SWIG_2="""%include <CA/CAProbabilityFunction.h>"""
        except LookupError,e:            
            PROBABILITY_FUNCTION_SWIG_1=''        
            PROBABILITY_FUNCTION_SWIG_2=''
        
        replaceLabelList.append(['PROBABILITY_FUNCTION_SWIG_1',PROBABILITY_FUNCTION_SWIG_1]) 
        replaceLabelList.append(['PROBABILITY_FUNCTION_SWIG_2',PROBABILITY_FUNCTION_SWIG_2]) 
        
        try:
            _features['LatticeMonitor']                        
            LATTICE_MONITOR_SWIG_1="""#include <CA/CACellStackFieldChangeWatcher.h>"""                        
            LATTICE_MONITOR_SWIG_2="""%include <CA/CACellStackFieldChangeWatcher.h>"""                        
        except LookupError,e:            
            LATTICE_MONITOR_SWIG_1=''        
            LATTICE_MONITOR_SWIG_2=''        
            
        replaceLabelList.append(['LATTICE_MONITOR_SWIG_1',LATTICE_MONITOR_SWIG_1]) 
        replaceLabelList.append(['LATTICE_MONITOR_SWIG_2',LATTICE_MONITOR_SWIG_2])             

        try:
            _features['Steppable']                        
            STEPPABLE_SWIG_1="""#include <CA/CASteppable.h>"""                        
            STEPPABLE_SWIG_2="""%include <CA/CASteppable.h>"""                        
        except LookupError,e:            
            STEPPABLE_SWIG_1=''        
            STEPPABLE_SWIG_2=''        
            
        replaceLabelList.append(['STEPPABLE_SWIG_1',STEPPABLE_SWIG_1]) 
        replaceLabelList.append(['STEPPABLE_SWIG_2',STEPPABLE_SWIG_2])             

        
        for replaceItems in replaceLabelList:
            moduleSwigText=re.sub(replaceItems[0],replaceItems[1], moduleSwigText)
            
        return moduleSwigText        
        
                
    def generateCMakeFile(self,_features={}):
        
        if 'Plugin' in _features.keys():
            text=''
            if _features['codeLayout']=='developerzone':
                text=self.cppTemplatesDict["CMakePluginDeveloperZone"]
            else:
                text=self.cppTemplatesDict["CMakePlugin"]
            pluginName=_features['Plugin']
            text=re.sub("PLUGIN_CORE_NAME",pluginName,text)     
            try:
                extraAttribFlag=_features['ExtraAttribute']
                text=re.sub("EXTRA_ATTRIBUTE_HEADER",pluginName+'Data.h',text)                     
            except LookupError,e:
                text=re.sub("EXTRA_ATTRIBUTE_HEADER",'',text)                     
            return text    
                
        else: # steppable
            return ''

            
            
    def generatePluginProxyFile(self,_features={}):
    
        replaceLabelList=[]
        pluginName=_features['Plugin']
        
        pluginProxyText=self.cppTemplatesDict["PluginProxy"]
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        pluginNameVar=lowerFirst(pluginName)
        
        PLUGIN_NAME_CORE=pluginName
        replaceLabelList.append(['PLUGIN_NAME_CORE',PLUGIN_NAME_CORE])
        
        PLUGIN_NAME_VAR=pluginNameVar
        replaceLabelList.append(['PLUGIN_NAME_VAR',PLUGIN_NAME_VAR])
        
        for replaceItems in replaceLabelList:
            pluginProxyText=re.sub(replaceItems[0],replaceItems[1], pluginProxyText)
            
        return pluginProxyText
        
    def generatePluginDLLSpecifier(self,_features={}):
    
        replaceLabelList=[]
        pluginName=_features['Plugin']
                
        pluginDLLSpecifierText=self.cppTemplatesDict["PluginDLLSpecifier"]
        
        PLUGIN_NAME_CORE=pluginName
        replaceLabelList.append(['PLUGIN_NAME_CORE',PLUGIN_NAME_CORE])
        
        PLUGIN_NAME_CAPITALS=pluginName.upper()
        replaceLabelList.append(['PLUGIN_NAME_CAPITALS',PLUGIN_NAME_CAPITALS])
        
        for replaceItems in replaceLabelList:
            pluginDLLSpecifierText=re.sub(replaceItems[0],replaceItems[1], pluginDLLSpecifierText)
            
        return pluginDLLSpecifierText        
        
    def generatePluginExtraAttributeFile(self,_features={}):
    
        replaceLabelList=[]
        pluginName=_features['Plugin']
        
        pluginExtraAttribText=self.cppTemplatesDict["PluginExtraAttributeData"]
        
        IFDEFLABEL=(pluginName+'Pata').upper()+'_H'
        replaceLabelList.append(['IFDEFLABEL',IFDEFLABEL])
        
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        pluginNameVar=lowerFirst(pluginName)
        
        PLUGIN_NAME_CORE=pluginName
        replaceLabelList.append(['PLUGIN_NAME_CORE',PLUGIN_NAME_CORE])
        
        PLUGIN_NAME_VAR=pluginNameVar
        replaceLabelList.append(['PLUGIN_NAME_VAR',PLUGIN_NAME_VAR])
        
        DLL_SPECIFIER_EXPORT=pluginName.upper()+"_EXPORT"
        replaceLabelList.append(['DLL_SPECIFIER_EXPORT',DLL_SPECIFIER_EXPORT])            
        
        
        for replaceItems in replaceLabelList:
            pluginExtraAttribText=re.sub(replaceItems[0],replaceItems[1], pluginExtraAttribText)
            
        return pluginExtraAttribText

    def generatePluginImplementationFile(self,_features={}):
    
        replaceLabelList=[]
        pluginName=_features['Plugin']
        
        pluginImplementationText=self.cppTemplatesDict["PluginImplementation"]
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        pluginNameVar=lowerFirst(pluginName)
        
        PLUGIN_NAME_CORE=pluginName
        replaceLabelList.append(['PLUGIN_NAME_CORE',PLUGIN_NAME_CORE])
        
        PLUGIN_NAME_VAR=pluginNameVar
        replaceLabelList.append(['PLUGIN_NAME_VAR',PLUGIN_NAME_VAR])
                       
        try:
            _features['ExtraAttribute']
            REGISTER_EXTRA_ATTRIBUTE='potts->getCellFactoryGroupPtr()->registerClass(&'+pluginNameVar+'DataAccessor);'
		
        except LookupError,e:
            REGISTER_EXTRA_ATTRIBUTE=''
                        
        replaceLabelList.append(['REGISTER_EXTRA_ATTRIBUTE',REGISTER_EXTRA_ATTRIBUTE])
        
        try:
            _features['EnergyFunction']
            
            REGISTER_ENERGY_FUNCTION='potts->registerEnergyFunctionWithName(this,\"'+pluginName+'\");'   
            ENERGY_FUNCTION_IMPLEMENTATION="""
double PLUGIN_NAME_COREPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {	
    double energy = 0;    if (oldCell){        //PUT YOUR CODE HERE    }else{        //PUT YOUR CODE HERE    }        if(newCell){        //PUT YOUR CODE HERE    }else{        //PUT YOUR CODE HERE    }        return energy;    }            
"""
            ENERGY_FUNCTION_IMPLEMENTATION=re.sub('PLUGIN_NAME_CORE',PLUGIN_NAME_CORE, ENERGY_FUNCTION_IMPLEMENTATION)
        except LookupError,e:
            REGISTER_ENERGY_FUNCTION=''
            ENERGY_FUNCTION_IMPLEMENTATION=''
        
        replaceLabelList.append(['REGISTER_ENERGY_FUNCTION',REGISTER_ENERGY_FUNCTION])
        replaceLabelList.append(['ENERGY_FUNCTION_IMPLEMENTATION',ENERGY_FUNCTION_IMPLEMENTATION])

        try:
            _features['LatticeMonitor']            
            REGISTER_LATTICE_MONITOR='potts->registerCellGChangeWatcher(this);'
            LATTICE_MONITOR_IMPLEMENTATION="""            
void PLUGIN_NAME_COREPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) 
{
        //This function will be called after each succesful pixel copy - field3DChange does usuall ohusekeeping tasks to make sure state of cells, and state of the lattice is uptdate    if (newCell){        //PUT YOUR CODE HERE    }else{        //PUT YOUR CODE HERE    }    if (oldCell){        //PUT YOUR CODE HERE    }else{        //PUT YOUR CODE HERE    }
		
}
"""            
            LATTICE_MONITOR_IMPLEMENTATION=re.sub('PLUGIN_NAME_CORE',PLUGIN_NAME_CORE, LATTICE_MONITOR_IMPLEMENTATION)
        except LookupError,e:
            REGISTER_LATTICE_MONITOR =''
            LATTICE_MONITOR_IMPLEMENTATION=''
            
        replaceLabelList.append(['REGISTER_LATTICE_MONITOR',REGISTER_LATTICE_MONITOR])            
        replaceLabelList.append(['LATTICE_MONITOR_IMPLEMENTATION',LATTICE_MONITOR_IMPLEMENTATION])            
            
        try:
            _features['Stepper']
            REGISTER_STEPPER = 'potts->registerStepper(this);'            
            STEPPER_IMPLEMENTATION="""
void PLUGIN_NAME_COREPlugin::step() {    //Put your code here - it will be invoked after every succesful pixel copy and after all lattice monitor finished running    	}
""" 
            STEPPER_IMPLEMENTATION=re.sub('PLUGIN_NAME_CORE',PLUGIN_NAME_CORE, STEPPER_IMPLEMENTATION)
        except LookupError,e:
            REGISTER_STEPPER=''
            STEPPER_IMPLEMENTATION=''
            
        replaceLabelList.append(['REGISTER_STEPPER',REGISTER_STEPPER])            
        replaceLabelList.append(['STEPPER_IMPLEMENTATION',STEPPER_IMPLEMENTATION])            
        

        for replaceItems in replaceLabelList:
            pluginImplementationText=re.sub(replaceItems[0],replaceItems[1], pluginImplementationText)
            
        return pluginImplementationText
        
        
    def generatePluginHeaderFile(self,_features={}):
    
        replaceLabelList=[]
        pluginName=_features['Plugin']
        
        pluginHeaderText=self.cppTemplatesDict["PluginHeader"]
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        pluginNameVar=lowerFirst(pluginName)
        
        PLUGIN_NAME_CORE=pluginName
        replaceLabelList.append(['PLUGIN_NAME_CORE',PLUGIN_NAME_CORE])
        
        PLUGIN_NAME_VAR=pluginNameVar
        replaceLabelList.append(['PLUGIN_NAME_VAR',PLUGIN_NAME_VAR])
        
        
        
        # set text for replace labels
        IFDEFLABEL=(pluginName+'Plugin').upper()+'_H'
        replaceLabelList.append(['IFDEFLABEL',IFDEFLABEL])
               
        
        try:
            _features['ExtraAttribute']
            EXTRA_ATTRIB_INCLUDES='#include \"'+pluginName+'Data.h\"'
            # # # EXTRA_ATTRIB_INCLUDES+="""
# # # """

            EXTRA_ATTRIB_ACCESSOR_DEFINE='BasicClassAccessor<'+pluginName+'Data> '+pluginNameVar+'DataAccessor;'
        
            EXTRA_ATTRIB_ACCESSOR_GET_PTR='BasicClassAccessor<'+pluginName+'Data> * '+'get'+pluginName+'DataAccessorPtr(){return & '+pluginNameVar+'DataAccessor;}'
		


        except LookupError,e:
            EXTRA_ATTRIB_INCLUDES=''
            EXTRA_ATTRIB_ACCESSOR_DEFINE=''
            EXTRA_ATTRIB_ACCESSOR_GET_PTR=''
            
            
        replaceLabelList.append(['EXTRA_ATTRIB_INCLUDES',EXTRA_ATTRIB_INCLUDES])
        replaceLabelList.append(['EXTRA_ATTRIB_ACCESSOR_DEFINE',EXTRA_ATTRIB_ACCESSOR_DEFINE])
        replaceLabelList.append(['EXTRA_ATTRIB_ACCESSOR_GET_PTR',EXTRA_ATTRIB_ACCESSOR_GET_PTR])        
        
        
        try:
            _features['EnergyFunction']
            
            ENERGY_FUNCTION_INCLUDE='#include <CompuCell3D/Potts3D/EnergyFunction.h>'            
            ENERGY_FUNCTION_INTERFACE='//Energy function interface\n'
            ENERGY_FUNCTION_INTERFACE+='        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);'
            ENERGY_FUNCTION_BASE=',public EnergyFunction'
            
        except LookupError,e:
            ENERGY_FUNCTION_INCLUDE=''
            ENERGY_FUNCTION_INTERFACE=''
            ENERGY_FUNCTION_BASE=''
            
            
        replaceLabelList.append(['ENERGY_FUNCTION_INCLUDE',ENERGY_FUNCTION_INCLUDE])
        replaceLabelList.append(['ENERGY_FUNCTION_INTERFACE',ENERGY_FUNCTION_INTERFACE])
        replaceLabelList.append(['ENERGY_FUNCTION_BASE',ENERGY_FUNCTION_BASE])
        
        
        try:
            _features['LatticeMonitor']            
            LATTICE_MONITOR_INCLUDE='#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>'
            LATTICE_MONITOR_INTERFACE='// CellChangeWatcher interface\n'
            LATTICE_MONITOR_INTERFACE+='        virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);'
            LATTICE_MONITOR_BASE=',public CellGChangeWatcher'
        
            
        except LookupError,e:
            LATTICE_MONITOR_INCLUDE=''
            LATTICE_MONITOR_INTERFACE=''
            LATTICE_MONITOR_BASE=''
            
        replaceLabelList.append(['LATTICE_MONITOR_INCLUDE',LATTICE_MONITOR_INCLUDE])            
        replaceLabelList.append(['LATTICE_MONITOR_INTERFACE',LATTICE_MONITOR_INTERFACE])            
        replaceLabelList.append(['LATTICE_MONITOR_BASE',LATTICE_MONITOR_BASE])   
            
        try:
            _features['Stepper']
                        
            STEPPER_INCLUDE='#include <CompuCell3D/Potts3D/Stepper.h>'
            STEPPER_INTERFACE='// Stepper interface\n'        
            STEPPER_INTERFACE+='        virtual void step();'        
            STEPPER_BASE=',public Stepper'
            
            
        except LookupError,e:
            STEPPER_INCLUDE=''
            STEPPER_INTERFACE=''
            STEPPER_BASE=''
            
        replaceLabelList.append(['STEPPER_INCLUDE',STEPPER_INCLUDE])            
        replaceLabelList.append(['STEPPER_INTERFACE',STEPPER_INTERFACE])            
        replaceLabelList.append(['STEPPER_BASE',STEPPER_BASE])                

        DLL_SPECIFIER_INCLUDE='#include \"'+pluginName+'DLLSpecifier.h\"'
        replaceLabelList.append(['DLL_SPECIFIER_INCLUDE',DLL_SPECIFIER_INCLUDE])            
        

        DLL_SPECIFIER_EXPORT=pluginName.upper()+"_EXPORT"
        replaceLabelList.append(['DLL_SPECIFIER_EXPORT',DLL_SPECIFIER_EXPORT])            
        
        # replacing labels with generated text
        for replaceItems in replaceLabelList:
            pluginHeaderText=re.sub(replaceItems[0],replaceItems[1], pluginHeaderText)
            
            
        return pluginHeaderText
        
# STEPPABLE GENERATOR

    # def generateCMakeFileSteppable(self,_features={}):
        
        # if 'Steppable' in _features.keys():
            # text=self.cppTemplatesDict["CMakeSteppable"]
            # moduleName=_features['Steppable']
            # text=re.sub("STEPPABLE_NAME_CORE",moduleName,text)       
            # return text    
                
        # else: # steppable
            # return ''


    def generateCMakeFileSteppable(self,_features={}):
                    
        if 'Steppable' in _features.keys():
            text=''
            if _features['codeLayout']=='developerzone':
                text=self.cppTemplatesDict["CMakeSteppableDeveloperZone"]
            else:
                text=self.cppTemplatesDict["CMakeSteppable"]
            steppableName=_features['Steppable']
            print 'text=',text
            text=re.sub("STEPPABLE_NAME_CORE",steppableName,text)     
            try:
                extraAttribFlag=_features['ExtraAttribute']
                text=re.sub("EXTRA_ATTRIBUTE_HEADER",steppableName+'Data.h',text)                     
            except LookupError,e:
                text=re.sub("EXTRA_ATTRIBUTE_HEADER",'',text)        
            
            print 'CMAKE TEXT=',text    
            return text    
                
        else: # steppable
            return ''

            
    def generateSteppableProxyFile(self,_features={}):
    
        replaceLabelList=[]
        steppableName=_features['Steppable']
        
        steppableProxyText=self.cppTemplatesDict["SteppableProxy"]
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        steppableNameVar=lowerFirst(steppableName)
        
        STEPPABLE_NAME_CORE=steppableName
        replaceLabelList.append(['STEPPABLE_NAME_CORE',STEPPABLE_NAME_CORE])
        
        STEPPABLE_NAME_VAR=steppableNameVar
        replaceLabelList.append(['STEPPABLE_NAME_VAR',STEPPABLE_NAME_VAR])
        
        for replaceItems in replaceLabelList:
            steppableProxyText=re.sub(replaceItems[0],replaceItems[1], steppableProxyText)
            
        return steppableProxyText
        
    def generateSteppableDLLSpecifier(self,_features={}):
    
        replaceLabelList=[]
        steppableName=_features['Steppable']
                
        steppableDLLSpecifierText=self.cppTemplatesDict["SteppableDLLSpecifier"]
        
        STEPPABLE_NAME_CORE=steppableName
        replaceLabelList.append(['STEPPABLE_NAME_CORE',STEPPABLE_NAME_CORE])
        
        STEPPABLE_NAME_CAPITALS=steppableName.upper()
        replaceLabelList.append(['STEPPABLE_NAME_CAPITALS',STEPPABLE_NAME_CAPITALS])
        
        for replaceItems in replaceLabelList:
            steppableDLLSpecifierText=re.sub(replaceItems[0],replaceItems[1], steppableDLLSpecifierText)
            
        return steppableDLLSpecifierText        
        
    def generateSteppableHeaderFile(self,_features={}):
    
        replaceLabelList=[]
        steppableName=_features['Steppable']
        
        steppableHeaderText=self.cppTemplatesDict["SteppableHeader"]
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        steppableNameVar=lowerFirst(steppableName)
        
        STEPPABLE_NAME_CORE=steppableName
        replaceLabelList.append(['STEPPABLE_NAME_CORE',STEPPABLE_NAME_CORE])
        
        STEPPABLE_NAME_VAR=steppableNameVar
        replaceLabelList.append(['STEPPABLE_NAME_VAR',STEPPABLE_NAME_VAR])
        
        try:
            print '_features[ExtraAttribute]=',_features['ExtraAttribute']
            _features['ExtraAttribute']
            EXTRA_ATTRIB_INCLUDES='#include \"'+steppableName+'Data.h\"'
            # # # EXTRA_ATTRIB_INCLUDES+="""
# # # #include <BasicUtils/BasicClassAccessor.h>
# # # #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
# # # """

            EXTRA_ATTRIB_ACCESSOR_DEFINE='BasicClassAccessor<'+steppableName+'Data> '+steppableNameVar+'DataAccessor;'
        
            EXTRA_ATTRIB_ACCESSOR_GET_PTR='BasicClassAccessor<'+steppableName+'Data> * '+'get'+steppableName+'DataAccessorPtr(){return & '+steppableNameVar+'DataAccessor;}'
		


        except LookupError,e:
            EXTRA_ATTRIB_INCLUDES=''
            EXTRA_ATTRIB_ACCESSOR_DEFINE=''
            EXTRA_ATTRIB_ACCESSOR_GET_PTR=''
            
            
        replaceLabelList.append(['EXTRA_ATTRIB_INCLUDES',EXTRA_ATTRIB_INCLUDES])
        replaceLabelList.append(['EXTRA_ATTRIB_ACCESSOR_DEFINE',EXTRA_ATTRIB_ACCESSOR_DEFINE])
        replaceLabelList.append(['EXTRA_ATTRIB_ACCESSOR_GET_PTR',EXTRA_ATTRIB_ACCESSOR_GET_PTR])         
        
        
        # set text for replace labels
        IFDEFLABEL=(steppableName+'Steppable').upper()+'_H'
        replaceLabelList.append(['IFDEFLABEL',IFDEFLABEL])
               
        DLL_SPECIFIER_EXPORT=steppableName.upper()+"_EXPORT"
        replaceLabelList.append(['DLL_SPECIFIER_EXPORT',DLL_SPECIFIER_EXPORT])            
        
        # replacing labels with generated text
        for replaceItems in replaceLabelList:
            steppableHeaderText=re.sub(replaceItems[0],replaceItems[1], steppableHeaderText)
            
            
        return steppableHeaderText
        
    def generateSteppableImplementationFile(self,_features={}):
    
        replaceLabelList=[]
        steppableName=_features['Steppable']
        
        steppableImplementationText=self.cppTemplatesDict["SteppableImplementation"]
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        steppableNameVar=lowerFirst(steppableName)
        
        STEPPABLE_NAME_CORE=steppableName
        replaceLabelList.append(['STEPPABLE_NAME_CORE',STEPPABLE_NAME_CORE])
        
        STEPPABLE_NAME_VAR=steppableNameVar
        replaceLabelList.append(['STEPPABLE_NAME_VAR',STEPPABLE_NAME_VAR])

        try:
            _features['ExtraAttribute']
            REGISTER_EXTRA_ATTRIBUTE='potts->getCellFactoryGroupPtr()->registerClass(&'+steppableNameVar+'DataAccessor);'
		
        except LookupError,e:
            REGISTER_EXTRA_ATTRIBUTE=''
                        
        replaceLabelList.append(['REGISTER_EXTRA_ATTRIBUTE',REGISTER_EXTRA_ATTRIBUTE])        

        for replaceItems in replaceLabelList:
            steppableImplementationText=re.sub(replaceItems[0],replaceItems[1], steppableImplementationText)
            
        return steppableImplementationText
        
    # # # def  generateSteppableRegistrationCode(self,_steppableName="GenericSteppable",_frequency=1,_steppableFile="",_indentationLevel=0,_indentationWidth=4):
        # # # try:
            # # # text=self.steppableTemplatesDict["SteppableRegistrationCode"]
        # # # except LookupError,e:
            # # # return ""
        
        # # # text=re.sub("STEPPABLENAME",_steppableName,text)        
        # # # text=re.sub("STEPPABLEFILE",_steppableFile,text)                
        # # # text=re.sub("FREQUENCY",str(_frequency),text)
        
        # # # # possible indentation of registration code - quite unlikely it wiil be needed
        # # # if _indentationLevel<0:
            # # # _indentationLevel=0
        
        # # # textLines=text.splitlines(True)
        
        # # # for i in range(len(textLines)):
            # # # textLines[i]=' '*_indentationWidth*_indentationLevel+textLines[i]
        
        # # # text=''.join(textLines)
        
        # # # return text
        
    def generateSteppableExtraAttributeFile(self,_features={}):
    
        replaceLabelList=[]
        steppableName=_features['Steppable']
        
        steppableExtraAttribText=self.cppTemplatesDict["SteppableExtraAttributeData"]
        
        IFDEFLABEL=(steppableName+'Pata').upper()+'_H'
        replaceLabelList.append(['IFDEFLABEL',IFDEFLABEL])
        
        
        lowerFirst = lambda s: s[:1].lower() + s[1:] if s else ''
        steppableNameVar=lowerFirst(steppableName)
        
        STEPPABLE_NAME_CORE=steppableName
        replaceLabelList.append(['STEPPABLE_NAME_CORE',STEPPABLE_NAME_CORE])
        
        STEPPABLE_NAME_VAR=steppableNameVar
        replaceLabelList.append(['STEPPABLE_NAME_VAR',STEPPABLE_NAME_VAR])
        
        DLL_SPECIFIER_EXPORT=steppableName.upper()+"_EXPORT"
        replaceLabelList.append(['DLL_SPECIFIER_EXPORT',DLL_SPECIFIER_EXPORT])            
        
        
        for replaceItems in replaceLabelList:
            steppableExtraAttribText=re.sub(replaceItems[0],replaceItems[1], steppableExtraAttribText)
            
        return steppableExtraAttribText
        
    
    def initCppTemplates(self):
        self.cppTemplatesDict["CMakePlugin"]="""
ADD_COMPUCELL3D_PLUGIN(PLUGIN_CORE_NAME LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})
"""    

        self.cppTemplatesDict["CMakePluginDeveloperZone"]="""
ADD_COMPUCELL3D_PLUGIN(PLUGIN_CORE_NAME LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})        
"""    


        self.cppTemplatesDict["PluginHeader"]="""
#ifndef IFDEFLABEL
#define IFDEFLABEL

#include <CompuCell3D/CC3D.h>
EXTRA_ATTRIB_INCLUDES

DLL_SPECIFIER_INCLUDE

class CC3DXMLElement;

namespace CompuCell3D {
    class Simulator;

    class Potts3D;
    class Automaton;
    //class AdhesionFlexData;
    class BoundaryStrategy;
    class ParallelUtilsOpenMP;
    
    template <class T> class Field3D;
    template <class T> class WatchableField3D;

    class DLL_SPECIFIER_EXPORT  PLUGIN_NAME_COREPlugin : public Plugin ENERGY_FUNCTION_BASE LATTICE_MONITOR_BASE STEPPER_BASE{
        
    private:    
        EXTRA_ATTRIB_ACCESSOR_DEFINE                
        CC3DXMLElement *xmlData;        
        
        Potts3D *potts;
        
        Simulator *sim;
        
        ParallelUtilsOpenMP *pUtils;            
        
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;        

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;
        WatchableField3D<CellG *> *cellFieldG;
        
    public:

        PLUGIN_NAME_COREPlugin();
        virtual ~PLUGIN_NAME_COREPlugin();
        
        EXTRA_ATTRIB_ACCESSOR_GET_PTR                

        
        ENERGY_FUNCTION_INTERFACE        
        LATTICE_MONITOR_INTERFACE
        STEPPER_INTERFACE        
        
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

        virtual void extraInit(Simulator *simulator);

        //Steerrable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
        virtual std::string steerableName();
        virtual std::string toString();

    };
};
#endif
        
"""
        self.cppTemplatesDict["PluginProxy"]="""        
#include "PLUGIN_NAME_COREPlugin.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Plugin, PLUGIN_NAME_COREPlugin>
PLUGIN_NAME_VARProxy("PLUGIN_NAME_CORE", "Autogenerated plugin - the author of the plugin should provide brief description here",
         &Simulator::pluginManager);
"""
        self.cppTemplatesDict["PluginExtraAttributeData"]="""
#ifndef IFDEFLABEL
#define IFDEFLABEL


#include <vector>
#include "PLUGIN_NAME_COREDLLSpecifier.h"

namespace CompuCell3D {

   
   class DLL_SPECIFIER_EXPORT PLUGIN_NAME_COREData{
      public:
         PLUGIN_NAME_COREData(){};
         
         ~PLUGIN_NAME_COREData(){};
         std::vector<float> array;
         int x;
         
         
   };
};
#endif
"""

        self.cppTemplatesDict["PluginDLLSpecifier"]="""
#ifndef PLUGIN_NAME_CAPITALS_EXPORT_H
#define PLUGIN_NAME_CAPITALS_EXPORT_H

    #if defined(_WIN32)
      #ifdef PLUGIN_NAME_COREShared_EXPORTS
          #define PLUGIN_NAME_CAPITALS_EXPORT __declspec(dllexport)
          #define PLUGIN_NAME_CAPITALS_EXPIMP_TEMPLATE
      #else
          #define PLUGIN_NAME_CAPITALS_EXPORT __declspec(dllimport)
          #define PLUGIN_NAME_CAPITALS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PLUGIN_NAME_CAPITALS_EXPORT
         #define PLUGIN_NAME_CAPITALS_EXPIMP_TEMPLATE
    #endif

#endif
"""
        self.cppTemplatesDict["PluginImplementation"]="""
#include <CompuCell3D/CC3D.h>        
using namespace CompuCell3D;

#include "PLUGIN_NAME_COREPlugin.h"


PLUGIN_NAME_COREPlugin::PLUGIN_NAME_COREPlugin():
pUtils(0),
lockPtr(0),
xmlData(0) ,
cellFieldG(0),
boundaryStrategy(0)
{}

PLUGIN_NAME_COREPlugin::~PLUGIN_NAME_COREPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr=0;
}

void PLUGIN_NAME_COREPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData=_xmlData;
    sim=simulator;
    potts=simulator->getPotts();
    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    
    pUtils=sim->getParallelUtils();
    lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr); 
   
   update(xmlData,true);
   
    REGISTER_EXTRA_ATTRIBUTE
    REGISTER_ENERGY_FUNCTION
    REGISTER_LATTICE_MONITOR    
    REGISTER_STEPPER
    
    simulator->registerSteerableObject(this);
}

void PLUGIN_NAME_COREPlugin::extraInit(Simulator *simulator){
    
}

LATTICE_MONITOR_IMPLEMENTATION
STEPPER_IMPLEMENTATION
ENERGY_FUNCTION_IMPLEMENTATION

void PLUGIN_NAME_COREPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
   set<unsigned char> cellTypesSet;

    CC3DXMLElement * exampleXMLElem=_xmlData->getFirstElement("Example");
    if (exampleXMLElem){
        double param=exampleXMLElem->getDouble();
        cerr<<"param="<<param<<endl;
        if(exampleXMLElem->findAttribute("Type")){
            std::string attrib=exampleXMLElem->getAttribute("Type");
            // double attrib=exampleXMLElem->getAttributeAsDouble("Type"); //in case attribute is of type double
            cerr<<"attrib="<<attrib<<endl;
        }
    }
    
    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy=BoundaryStrategy::getInstance();

}


std::string PLUGIN_NAME_COREPlugin::toString(){
    return "PLUGIN_NAME_CORE";
}


std::string PLUGIN_NAME_COREPlugin::steerableName(){
    return toString();
}
"""
        self.cppTemplatesDict["CMakeSteppable"]="""
ADD_COMPUCELL3D_STEPPABLE(STEPPABLE_NAME_CORE   LINK_LIBRARIES ${STEPPABLE_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})        
"""        

        self.cppTemplatesDict["CMakeSteppableDeveloperZone"]="""
ADD_COMPUCELL3D_STEPPABLE(STEPPABLE_NAME_CORE   LINK_LIBRARIES ${STEPPABLE_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})            
"""    



        self.cppTemplatesDict["SteppableProxy"]="""
#include "STEPPABLE_NAME_CORE.h"

#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <BasicUtils/BasicPluginProxy.h>

BasicPluginProxy<Steppable, STEPPABLE_NAME_CORE> 
STEPPABLE_NAME_VARProxy("STEPPABLE_NAME_CORE", "Autogenerated steppeble - the author of the plugin should provide brief description here",
	    &Simulator::steppableManager);        
"""        
    
        self.cppTemplatesDict["SteppableDLLSpecifier"]="""
#ifndef STEPPABLE_NAME_CAPITALS_EXPORT_H
#define STEPPABLE_NAME_CAPITALS_EXPORT_H

    #if defined(_WIN32)
      #ifdef STEPPABLE_NAME_COREShared_EXPORTS
          #define STEPPABLE_NAME_CAPITALS_EXPORT __declspec(dllexport)
          #define STEPPABLE_NAME_CAPITALS_EXPIMP_TEMPLATE
      #else
          #define STEPPABLE_NAME_CAPITALS_EXPORT __declspec(dllimport)
          #define STEPPABLE_NAME_CAPITALS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define STEPPABLE_NAME_CAPITALS_EXPORT
         #define STEPPABLE_NAME_CAPITALS_EXPIMP_TEMPLATE
    #endif

#endif
"""

        self.cppTemplatesDict["SteppableHeader"]="""
#ifndef IFDEFLABEL
#define IFDEFLABEL

#include <CompuCell3D/CC3D.h>

EXTRA_ATTRIB_INCLUDES

#include "STEPPABLE_NAME_COREDLLSpecifier.h"


namespace CompuCell3D {
    
  template <class T> class Field3D;
  template <class T> class WatchableField3D;

    class Potts3D;
    class Automaton;
    class BoundaryStrategy;
    class CellInventory;
    class CellG;
  
  class DLL_SPECIFIER_EXPORT STEPPABLE_NAME_CORE : public Steppable {

    EXTRA_ATTRIB_ACCESSOR_DEFINE                
    WatchableField3D<CellG *> *cellFieldG;
    Simulator * sim;
    Potts3D *potts;
    CC3DXMLElement *xmlData;
    Automaton *automaton;
    BoundaryStrategy *boundaryStrategy;
    CellInventory * cellInventoryPtr;
    
    Dim3D fieldDim;

    
  public:
    STEPPABLE_NAME_CORE ();
    virtual ~STEPPABLE_NAME_CORE ();
    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
    virtual void extraInit(Simulator *simulator);

    EXTRA_ATTRIB_ACCESSOR_GET_PTR
    
    //steppable interface
    virtual void start();
    virtual void step(const unsigned int currentStep);
    virtual void finish() {}


    //SteerableObject interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
	 virtual std::string toString();

  };
};
#endif        
"""        
        self.cppTemplatesDict["SteppableImplementation"]="""

#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

using namespace std;

#include "STEPPABLE_NAME_CORE.h"

STEPPABLE_NAME_CORE::STEPPABLE_NAME_CORE() : cellFieldG(0),sim(0),potts(0),xmlData(0),boundaryStrategy(0),automaton(0),cellInventoryPtr(0){}

STEPPABLE_NAME_CORE::~STEPPABLE_NAME_CORE() {
}


void STEPPABLE_NAME_CORE::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
  xmlData=_xmlData;
  
  potts = simulator->getPotts();
  cellInventoryPtr=& potts->getCellInventory();
  sim=simulator;
  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
  fieldDim=cellFieldG->getDim();

  REGISTER_EXTRA_ATTRIBUTE
  simulator->registerSteerableObject(this);

  update(_xmlData,true);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void STEPPABLE_NAME_CORE::extraInit(Simulator *simulator){
    //PUT YOUR CODE HERE
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void STEPPABLE_NAME_CORE::start(){

  //PUT YOUR CODE HERE

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void STEPPABLE_NAME_CORE::step(const unsigned int currentStep){
    //REPLACE SAMPLE CODE BELOW WITH YOUR OWN
	CellInventory::cellInventoryIterator cInvItr;
	CellG * cell=0;
    
    cerr<<"currentStep="<<currentStep<<endl;
	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr )
	{
		cell=cellInventoryPtr->getCell(cInvItr);
        cerr<<"cell.id="<<cell->id<<" vol="<<cell->volume<<endl;
    }

}


void STEPPABLE_NAME_CORE::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
   set<unsigned char> cellTypesSet;

    CC3DXMLElement * exampleXMLElem=_xmlData->getFirstElement("Example");
    if (exampleXMLElem){
        double param=exampleXMLElem->getDouble();
        cerr<<"param="<<param<<endl;
        if(exampleXMLElem->findAttribute("Type")){
            std::string attrib=exampleXMLElem->getAttribute("Type");
            // double attrib=exampleXMLElem->getAttributeAsDouble("Type"); //in case attribute is of type double
            cerr<<"attrib="<<attrib<<endl;
        }
    }
    
    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy=BoundaryStrategy::getInstance();

}

std::string STEPPABLE_NAME_CORE::toString(){
   return "STEPPABLE_NAME_CORE";
}

std::string STEPPABLE_NAME_CORE::steerableName(){
   return toString();
}
        
"""        
        self.cppTemplatesDict["SteppableExtraAttributeData"]="""
#ifndef IFDEFLABEL
#define IFDEFLABEL


#include <vector>
#include "STEPPABLE_NAME_COREDLLSpecifier.h"

namespace CompuCell3D {

   
   class DLL_SPECIFIER_EXPORT STEPPABLE_NAME_COREData{
      public:
         STEPPABLE_NAME_COREData(){};
         
         ~STEPPABLE_NAME_COREData(){};
         std::vector<float> array;
         int x;
         
         
   };
};
#endif
"""

        self.cppTemplatesDict["CACMakeModule"]="""
ADD_CA_MODULE(MODULE_CORE_NAME LINK_LIBRARIES ${MODULE_LIBS} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS} PYTHON_WRAP)
"""    

        self.cppTemplatesDict["moduleDLLSpecifier"]="""
#ifndef MODULE_NAME_CAPITALS_EXPORT_H
#define MODULE_NAME_CAPITALS_EXPORT_H

    #if defined(_WIN32)
      #ifdef MODULE_NAME_COREShared_EXPORTS
          #define MODULE_NAME_CAPITALS_EXPORT __declspec(dllexport)
          #define MODULE_NAME_CAPITALS_EXPIMP_TEMPLATE
      #else
          #define MODULE_NAME_CAPITALS_EXPORT __declspec(dllimport)
          #define MODULE_NAME_CAPITALS_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define MODULE_NAME_CAPITALS_EXPORT
         #define MODULE_NAME_CAPITALS_EXPIMP_TEMPLATE
    #endif

#endif        
"""        

        self.cppTemplatesDict["CAModuleHeader"]="""
        
#ifndef IFDEFLABEL
#define IFDEFLABEL

#include <string>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Point3D.h>

PROBABILITY_FUNCTION_INCLUDE
LATTICE_MONITOR_INCLUDE
STEPPABLE_INCLUDE
DLL_SPECIFIER_INCLUDE


namespace CompuCell3D{

    //have to declare here all the classes that will be passed to this class from Python
    class CAManager;
    class Point3D;    
    class Dim3D;

    class DLL_SPECIFIER_EXPORT MODULE_NAME_CORE: PROBABILITY_FUNCTION_BASE LATTICE_MONITOR_BASE STEPPABLE_BASE{
        int carryingCapacity;
    public:
    
        MODULE_NAME_CORE();
        virtual ~MODULE_NAME_CORE();

        PROBABILITY_FUNCTION_INTERFACE        
        LATTICE_MONITOR_INTERFACE        
        STEPPABLE_INTERFACE
        
        //CA SimulationObject interface
        virtual void init(CAManager *_caManager);
        virtual void extraInit(){} //implement if necessary
        virtual void extraInit2(){} //implement if necessary       
        virtual std::string toString();
        


    
    };
};

#endif

"""

        self.cppTemplatesDict["CAModuleImplementation"]="""
#include <CA/CACell.h>
#include <CA/CAManager.h>
#include <CA/CACellStack.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>

#include "MODULE_NAME_CORE.h"

//#define _DEBUG
using namespace CompuCell3D;
using namespace std;

MODULE_NAME_CORE::MODULE_NAME_CORE():
carryingCapacity(1)
{
}

//////////////////////////////////////////////////////////////////////////////////////////

MODULE_NAME_CORE::~MODULE_NAME_CORE(){}

//////////////////////////////////////////////////////////////////////////////////////////

void MODULE_NAME_CORE::init(CAManager *_caManager){
    RUNTIME_ASSERT_OR_THROW("CellTrail::init _caManager cannot be NULL!",_caManager);
    caManager = _caManager;
    cellFieldS = caManager->getCellFieldS();
    fieldDim = cellFieldS->getDim();
	carryingCapacity = caManager ->getCellCarryingCapacity();

}

//////////////////////////////////////////////////////////////////////////////////////////
std::string MODULE_NAME_CORE::toString(){return "MODULE_NAME_CORE";}

PROBABILITY_FUNCTION_IMPLEMENTATION
LATTICE_MONITOR_IMPLEMENTATION
STEPPABLE_IMPLEMENTATION

"""
        
        self.cppTemplatesDict["CAModuleSwig"]="""
// -*-c++-*-


%module ("threads"=1) MODULE_NAME_CORE

%include "typemaps.i"

%include stl.i //to ensure stl functionality 


// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_set.i"

// C++ std::map handling
%include "std_vector.i"

%include "stl.i"

%import "../../CAPython/CoreObjects/CoreObjects.i"


// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.
//DOCSTRINGS



%include <windows.i>

%{

PROBABILITY_FUNCTION_SWIG_1
LATTICE_MONITOR_SWIG_1
STEPPABLE_SWIG_1
#include <CA/modules/MODULE_NAME_CORE/MODULE_NAME_CORE.h>



// Namespaces
using namespace std;
using namespace CompuCell3D;



%}




%include stl.i //to ensure stl functionality 

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_set.i"

// C++ std::map handling
%include "std_vector.i"

%include "stl.i"

//enables better handling of STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

// %include "swig_includes/numpy.i"
// // // %include "pyinterface/swig_includes/numpy.i"

// // // %init %{
    // // // import_array();
// // // %}


//C arrays
//%include "carrays.i"

// ******************************
// Third Party Classes
// ******************************
#define DLL_SPECIFIER_EXPORT

PROBABILITY_FUNCTION_SWIG_2
LATTICE_MONITOR_SWIG_2
STEPPABLE_SWIG_2
%include <CA/modules/MODULE_NAME_CORE/MODULE_NAME_CORE.h>

%extend CompuCell3D::MODULE_NAME_CORE{
      %pythoncode %{

    def moduleExtraPyFunction(self,*args,**kwds):
        print "HERE YOU IMPLEMENT ALL ADDITIONAL PYTHON FUNCTIONS TO BE USED IN THE MODULE. E.G. FUNCTION THAT CONFIGURES THE MODULE"

	%}


};
"""


