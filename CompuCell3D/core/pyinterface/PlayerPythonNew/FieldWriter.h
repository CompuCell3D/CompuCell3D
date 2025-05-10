#ifndef FIELDWRITER_H
#define FIELDWRITER_H

#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <Utils/Coordinates3D.h>
#include "FieldStorage.h"

#include <CompuCell3D/Potts3D/Cell.h>


#include "FieldExtractorDLLSpecifier.h"
#include <typeindex>

class FieldStorage;
//class vtkIntArray;
//class vtkDoubleArray;
//class vtkFloatArray;
//class vtkPoints;
//class vtkCellArray;
//class vtkObject;
class vtkStructuredPoints;



namespace CompuCell3D{

	//have to declare here all the classes that will be passed to this class from Python
	class Potts3D;
	class Simulator;
	class Dim3D;
    template <class T> class Field3D;

	class FIELDEXTRACTOR_EXPORT  FieldWriter{
	public:
		Potts3D * potts;
		Simulator *sim;
		FieldWriter();
		~FieldWriter();

		void setFieldStorage(FieldStorage * _fsPtr){fsPtr=_fsPtr;}
		FieldStorage * getFieldStorage(FieldStorage * _fsPtr){return fsPtr;}
      void init(Simulator * _sim);

		void setFileTypeToBinary(bool flag);  // not currently used
		void addCellFieldForOutput();
		bool addConFieldForOutput(std::string _conFieldName);
		bool addScalarFieldForOutput(std::string _scalarFieldName);
		bool addScalarFieldCellLevelForOutput(std::string _scalarFieldCellLevelName);
		bool addVectorFieldForOutput(std::string _vectorFieldName);
		bool addVectorFieldCellLevelForOutput(std::string _vectorFieldCellLevelName);

		void clear();
		void writeFields(std::string _fileName);

		void generatePIFFileFromVTKOutput(std::string _vtkFileName,std::string _pifFileName,short _dimX, short _dimY, short _dimZ, std::map<int,std::string> &typeIdTypeNameMap);
	    void generatePIFFileFromCurrentStateOfSimulation(std::string _pifFileName);

	private:
		FieldStorage * fsPtr;
		std::vector<Coordinates3D<double> > hexagonVertices;
        vtkStructuredPoints *latticeData;
		std::vector<std::string> arrayNameVec;
		bool binaryFlag;   // not currently used
		/*Dim3D fieldDim;*/

        typedef std::unordered_map<std::type_index, std::function<bool(const std::string &, void *)>> concentrationFunctionMap_t;

        template<typename T>
        bool serializeConcentrationFieldTyped(const std::string & _conFieldName , Field3D<T> *conFieldPtr);

        void initializeSerializeConcentrationFunctionMap();

        std::tuple<std::type_index, void *> getFieldTypeAndPointer(const std::string &fieldName);

        concentrationFunctionMap_t serializeConcentrationFunctionMap;

	};


};



#endif
