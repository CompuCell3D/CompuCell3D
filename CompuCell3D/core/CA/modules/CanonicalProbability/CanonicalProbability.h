#ifndef CANONICALPROBABILITY_H
#define CANONICALPROBABILITY_H

#include <string>
#include <CA/ProbabilityFunction.h>
#include "CanonicalProbabilityDLLSpecifier.h"


namespace CompuCell3D{

	//have to declare here all the classes that will be passed to this class from Python
	class CAManager;
    class Point3D;    
	class Dim3D;

	class CANONICALPROBABILITY_EXPORT CanonicalProbability: public ProbabilityFunction{
		int carryingCapacity;
	public:
    
        CanonicalProbability();
        virtual ~CanonicalProbability();
        //local API
		float diffCoeff;
		float deltaT;



        //ProbabilityFunction interface
        virtual void init(CAManager *_caManager);
        virtual std::string toString();
        virtual float calculate(const Point3D & _source, const Point3D & _target);    


    
	};
};

#endif
