#ifndef CHEMOTAXISPROBABILITY_H
#define CHEMOTAXISPROBABILITY_H

#include <string>
#include <vector>
#include <CA/ProbabilityFunction.h>
#include "ChemotaxisProbabilityDLLSpecifier.h"


namespace CompuCell3D{

	//have to declare here all the classes that will be passed to this class from Python
	class CAManager;
    class Point3D;    
	class Dim3D;

	template<typename T>
	class Field3D;

	class CHEMOTAXISPROBABILITY_EXPORT ChemotaxisData{
	public:
		ChemotaxisData():
		lambda(0.0),
		type(0)		
		{}
		float lambda;
		unsigned char type;
		std::string typeName;
		Field3D<float> * concField;

	};

	class CHEMOTAXISPROBABILITY_EXPORT ChemotaxisProbability: public ProbabilityFunction{
		int carryingCapacity;
		//std::map<std::string,std::vector<ChemotaxisData> > chemotaxisDataMap;

		std::vector<std::vector<ChemotaxisData> > vecChemotaxisDataByType;
	public:
    
        ChemotaxisProbability();
        virtual ~ChemotaxisProbability();
        //local API
        void addChemotixisData(std::string _fieldName, std::string _typeName, float _lambda);
		void clearChemotaxisData();
		float diffCoeff;
		float deltaT;



        //ProbabilityFunction interface
        virtual void init(CAManager *_caManager);
        virtual std::string toString();
        virtual float calculate(const CACell * _sourceCell,const Point3D & _source, const Point3D & _target);    


    
	};
};

#endif
