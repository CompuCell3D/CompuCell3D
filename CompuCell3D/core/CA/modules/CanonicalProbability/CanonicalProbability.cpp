#include "CanonicalProbability.h"
#include <CA/CAManager.h>
#include <CA/CACellStack.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>

//#define _DEBUG
using namespace CompuCell3D;
using namespace std;

CanonicalProbability::CanonicalProbability():
diffCoeff(0.0),
deltaT(0.0),
carryingCapacity(1),
ProbabilityFunction()
{
}

//////////////////////////////////////////////////////////////////////////////////////////

CanonicalProbability::~CanonicalProbability(){}

//////////////////////////////////////////////////////////////////////////////////////////

void CanonicalProbability::init(CAManager *_caManager){
    caManager = _caManager;
    cellFieldS = caManager->getCellFieldS();
    fieldDim = cellFieldS->getDim();
	carryingCapacity = caManager ->getCellCarryingCapacity();

}

//////////////////////////////////////////////////////////////////////////////////////////
std::string CanonicalProbability::toString(){return "CanonicalProbability";}


float CanonicalProbability::calculate(const Point3D & _source, const Point3D & _target){
	
	


	CACellStack * sourceStack = cellFieldS -> get(_source);
	CACellStack * targetStack = cellFieldS -> get(_target);
	int targetFillLevel=0;
	
	if (targetStack ){
		targetFillLevel = targetStack -> getFillLevel() ;
#ifdef _DEBUG
		cerr<<"_source="<<_source<<" _target="<<_target<<endl;
		cerr<<"targetFillLevel ="<<targetFillLevel <<endl;
		cerr<<"carryingCapacity  = " <<carryingCapacity <<endl;
#endif

	}


	float prob = diffCoeff*deltaT/(2*((_source.x-_target.x)*(_source.x-_target.x)+(_source.y-_target.y)*(_source.y-_target.y)+(_source.z-_target.z)*(_source.z-_target.z)))
	*(carryingCapacity-targetFillLevel)/(float)(carryingCapacity);

	if (targetStack ){
		
#ifdef _DEBUG
		cerr<<"prob="<<prob<<endl;
#endif
	}

    return prob;
}    