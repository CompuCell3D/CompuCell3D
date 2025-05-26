#ifndef SERIALIZERDE_H
#define SERIALIZERDE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

#include <CompuCell3D/Field3D/Dim3D.h>

#include <SerializerDEDLLSpecifier.h>
#include <typeindex>


namespace CompuCell3D{

	//have to declare here all the classes that will be passed to this class from Python
	class Potts3D;
	class Simulator;	
	class CellG;
	template <class T> class Field3D;
	template <class T> class WatchableField3D;


	class SERIALIZERDE_EXPORT SerializeData{
	public:
		SerializeData():objectPtr(0){}
		std::string moduleName;
		std::string moduleType;
		std::string objectName;
		std::string objectType;
		std::string fileName;
		std::string fileFormat;
		void * objectPtr;
		std::string generateXMlStub(){return "";};

	};


	class SERIALIZERDE_EXPORT SerializerDE{
	public:

		SerializerDE();
		void init(Simulator * _sim);
		virtual ~SerializerDE();
		bool serializeConcentrationField(SerializeData &_sd);
//        bool serializeGenericConcentrationField(SerializeData &_sd);
		bool serializeCellField(SerializeData &_sd);
		bool serializeScalarField(SerializeData &_sd);
		bool serializeScalarFieldCellLevel(SerializeData &_sd);
        bool serializeSharedVectorFieldNumpy(SerializeData &_sd);

		bool serializeVectorField(SerializeData &_sd);
		bool serializeVectorFieldCellLevel(SerializeData &_sd);

		bool loadCellField(SerializeData &_sd);
		bool loadConcentrationField(SerializeData &_sd);
//        bool loadGenericConcentrationField(SerializeData &_sd);
		bool loadScalarField(SerializeData &_sd);
		bool loadScalarFieldCellLevel(SerializeData &_sd);
        bool loadSharedVectorFieldNumpy(SerializeData &_sd);
		bool loadVectorField(SerializeData &_sd);
		bool loadVectorFieldCellLevel(SerializeData &_sd);
		// virtual void readFromFile(){}

		// std::string fileName;
		// std::string auxPath;
		// std::string serializedFileExtension;
		std::vector<SerializeData> serializedDataVec;
	private:
		Dim3D fieldDim;
		Simulator *sim;
		Potts3D *potts;
		Field3D<CellG*> * cellFieldG;
        std::tuple<std::type_index, void *> getFieldTypeAndPointer(const std::string &fieldName);

        typedef std::unordered_map<std::type_index, std::function<bool(SerializeData &, void *)>> concentrationFunctionMap_t;

        template<typename T>
        bool serializeConcentrationFieldTyped(SerializeData &_sd, Field3D<T> *fieldPtr);

        template<typename T>
        bool loadConcentrationFieldTyped(SerializeData &_sd, Field3D<T> *fieldPtr);


        void initializeSerializeConcentrationFunctionMap();
        void initializeLoadConcentrationFunctionMap();

        concentrationFunctionMap_t serializeConcentrationFunctionMap;
        concentrationFunctionMap_t loadConcentrationFunctionMap;
	};

}


#endif