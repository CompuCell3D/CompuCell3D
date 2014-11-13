// -*-c++-*-


%module ("threads"=1) CAPDESolvers

//enables better handling of STL exceptions
%include "exception.i"
// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_set.i"

// C++ std::map handling
%include "std_vector.i"

%include "stl.i"

//%include "carrays.i"
//%array_class(int, intArray);
//%array_class(double, doubleArray);

%import "../../../CAPython/CoreObjects.i"
//%import "../CoreObjects.i"


%include "typemaps.i"

// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.
//DOCSTRINGS



%include <windows.i>

%{
// CompuCell3D Include Files
//#include <CompuCell3D/Field3D/Point3D.h>
//#include <CompuCell3D/Field3D/Dim3D.h>

#include <CompuCell3D/Field3D/Array3D.h>
#include <CA/modules/CAPDESolvers/CABoundaryConditionSpecifier.h>
#include <CA/modules/CAPDESolvers/DiffSecrData.h>
#include <CA/modules/CAPDESolvers/DiffusionSolverFE.h>




// Namespaces
using namespace std;
using namespace CompuCell3D;



%}




//////%include stl.i //to ensure stl functionality 
//////
//////// // // %include "CompuCellExtraIncludes.i"
//////
//////// C++ std::string handling
//////%include "std_string.i"
//////
//////// C++ std::map handling
//////%include "std_map.i"
//////
//////// C++ std::map handling
//////%include "std_set.i"
//////
//////// C++ std::map handling
//////%include "std_vector.i"
//////
//////%include "stl.i"
//////
////////enables better handling of STL exceptions
//////%include "exception.i"

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
#define CAPDESOLVERS_EXPORT

//%include <CompuCell3D/Field3D/Point3D.h>
//%include <CompuCell3D/Field3D/Dim3D.h>


%include <CompuCell3D/Field3D/Array3D.h>

//%template(Array3DContiguousFloat) CompuCell3D::Array3DContiguous<float>;

//////
//////%include <CA/modules/PDESolvers/DiffusableVectorCommon.h>
//////

//////%template(stdvectorstring) std::vector<std::string>;

//%ignore CompuCell3D::SecretionData::secretionConst;

//%ignore CompuCell3D::CABoundaryConditionSpecifier::BCType;
//%ignore CompuCell3D::CABoundaryConditionSpecifier::BCPosition;
//%ignore CompuCell3D::CABoundaryConditionSpecifier;

%include <CA/modules/CAPDESolvers/CABoundaryConditionSpecifier.h>


%include <CA/modules/CAPDESolvers/DiffSecrData.h>
%include <CA/modules/CAPDESolvers/DiffusionSolverFE.h>

%extend CompuCell3D::DiffusionSolverFE{
      %pythoncode %{
    def addFieldsPy(self,_fieldList):
        print '_fieldList=',_fieldList
        self.addFields(['FGF','VEGF'])            
	

    def addField(self,*args,**kwds):
        
        try:
            fieldName=kwds['Name']
            self.addDiffusionAndSecretionData(fieldName)
            diffData = self.getDiffusionData(fieldName)
            secrData = self.getSecretionData(fieldName)
            bcData = self.getBoundaryConditionData(fieldName)
        except LookupError:
            raise AttributeError('The field you define needs to have a name! Use "Name" as an argument of the "addField" function')
        
        diffDataPy=None
        secrDataPy=None
        try:
            diffDataPy=kwds['DiffusionData']
        except LookupError:
            pass 
                
        try:
            secrDataPy=kwds['SecretionData']
        except LookupError:
            pass 

        try:
            diffData.diffConst = diffDataPy['DiffusionConstant']
        except LookupError:
            pass 

        try:
            diffData.decayConst = diffDataPy['DecayConstant']
        except LookupError:
            pass 

        
        for typeName in secrDataPy.keys():
            
            secrData.setTypeNameSecrConst(typeName,secrDataPy[typeName])

        bcDict = None
        try:
            bcDict=kwds['BoundaryConditions']
        except LookupError:
            pass

        MIN_X=0;MAX_X=1;
        MIN_Y=2;MAX_Y=3;
        MIN_Z=4;MAX_Z=5;

        PERIODIC=0
        CONSTANT_VALUE=1
        CONSTANT_DERIVATIVE=2

        if bcDict is not None:
            if 'X' in bcDict.keys():
                xBCData = bcDict['X']
                if isinstance(xBCData, list):
                    for dataDict in xBCData:
                        if isinstance(dataDict, dict):
                            posIndex=-1
                            bcType=-1
                            value=0.0
                            try:
                                pos=dataDict['Position']                                
                                if pos=='Min':
                                    posIndex=MIN_X
                                if pos=='Max':
                                    posIndex=MAX_X

                            except:
                                raise AttributeError ('Could not find "Position" in the X axis boundary condition specification')                                

                            if 'ConstantValue' in dataDict.keys():
                                bcType=CONSTANT_VALUE
                                
                                value = float(dataDict['ConstantValue'])
                                

                            elif 'ConstantDerivative' in dataDict.keys():
                                bcType=CONSTANT_DERIVATIVE
                                value = float(dataDict['ConstantDerivative'])
                            else:
                                AttributeError ('Could not find "ConstantValue" or "ConstantValue" in the X axis boundary condition specification')
                            
                            if posIndex>=0 and bcType>=0:
                                bcData.setPlanePosition(posIndex,bcType)
                                bcData.setValues(posIndex,value)
                                #bcData.planePositions[posIndex]=bcType
                                #bcData.values[posIndex]=value

                elif xBCData == 'Periodic':
                    #bcData.planePositions[MIN_X]=PERIODIC
                    #bcData.planePositions[MAX_X]=PERIODIC
                    bcData.setPlanePosition(MIN_X,PERIODIC)
                    bcData.setPlanePosition(MAX_X,PERIODIC)


                else:
                    AttributeError ('Wrong specification of boundary conditions for X axis')



            if 'Y' in bcDict.keys():
                yBCData = bcDict['Y']
                if isinstance(yBCData, list):
                    for dataDict in yBCData:
                        if isinstance(dataDict, dict):
                            posIndex=-1
                            bcType=-1
                            value=0.0
                            try:
                                pos=dataDict['Position']                                
                                if pos=='Min':
                                    posIndex=MIN_Y
                                if pos=='Max':
                                    posIndex=MAX_Y

                            except:
                                raise AttributeError ('Could not find "Position" in the Y axis boundary condition specification')                                

                            if 'ConstantValue' in dataDict.keys():
                                bcType=CONSTANT_VALUE
                                
                                value = float(dataDict['ConstantValue'])
                                

                            elif 'ConstantDerivative' in dataDict.keys():
                                bcType=CONSTANT_DERIVATIVE
                                value = float(dataDict['ConstantDerivative'])
                            else:
                                AttributeError ('Could not find "ConstantValue" or "ConstantValue" in the X axis boundary condition specification')
                            
                            if posIndex>=0 and bcType>=0:
                                bcData.setPlanePosition(posIndex,bcType)
                                bcData.setValues(posIndex,value)
                                #bcData.planePositions[posIndex]=bcType
                                #bcData.values[posIndex]=value

                elif yBCData == 'Periodic':
                    #bcData.planePositions[MIN_Y]=PERIODIC
                    #bcData.planePositions[MAX_Y]=PERIODIC
                    bcData.setPlanePosition(MIN_Y,PERIODIC)
                    bcData.setPlanePosition(MAX_Y,PERIODIC)


                else:
                    AttributeError ('Wrong specification of boundary conditions for Y axis')

            if 'Z' in bcDict.keys():
                zBCData = bcDict['Z']
                if isinstance(zBCData, list):
                    for dataDict in zBCData:
                        if isinstance(dataDict, dict):
                            posIndex=-1
                            bcType=-1
                            value=0.0
                            try:
                                pos=dataDict['Position']                                
                                if pos=='Min':
                                    posIndex=MIN_Z
                                if pos=='Max':
                                    posIndex=MAX_Z

                            except:
                                raise AttributeError ('Could not find "Position" in the Z axis boundary condition specification')                                

                            if 'ConstantValue' in dataDict.keys():
                                bcType=CONSTANT_VALUE
                                
                                value = float(dataDict['ConstantValue'])
                                

                            elif 'ConstantDerivative' in dataDict.keys():
                                bcType=CONSTANT_DERIVATIVE
                                value = float(dataDict['ConstantDerivative'])
                            else:
                                AttributeError ('Could not find "ConstantValue" or "ConstantValue" in the Z axis boundary condition specification')
                            
                            if posIndex>=0 and bcType>=0:
                                bcData.setPlanePosition(posIndex,bcType)
                                bcData.setValues(posIndex,value)
                                #bcData.planePositions[posIndex]=bcType
                                #bcData.values[posIndex]=value

                elif zBCData == 'Periodic':
                    #bcData.planePositions[MIN_Z]=PERIODIC
                    #bcData.planePositions[MAX_Z]=PERIODIC
                    bcData.setPlanePosition(MIN_Z,PERIODIC)
                    bcData.setPlanePosition(MAX_Z,PERIODIC)

                else:
                    AttributeError ('Wrong specification of boundary conditions for Z axis')

	%}


};


