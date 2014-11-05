#include "ChemotaxisProbability.h"
#include <CA/CAManager.h>
#include <CA/CACell.h>
#include <CA/CACellStack.h>

#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <climits>

//#define _DEBUG
using namespace CompuCell3D;
using namespace std;

ChemotaxisProbability::ChemotaxisProbability():
diffCoeff(0.0),
deltaT(0.0),
carryingCapacity(1),
ProbabilityFunction()
{
	vecChemotaxisDataByType.assign(UCHAR_MAX+1,vector<ChemotaxisData>());
}

//////////////////////////////////////////////////////////////////////////////////////////

ChemotaxisProbability::~ChemotaxisProbability(){}

//////////////////////////////////////////////////////////////////////////////////////////

void ChemotaxisProbability::init(CAManager *_caManager){
    caManager = _caManager;
    cellFieldS = caManager->getCellFieldS();
    fieldDim = cellFieldS->getDim();
	carryingCapacity = caManager ->getCellCarryingCapacity();

}
//////////////////////////////////////////////////////////////////////////////////////////
void ChemotaxisProbability::addChemotixisData(std::string _fieldName, std::string _typeName, float _lambda){
	ChemotaxisData chemData;
	chemData.typeName=_typeName;
	chemData.type = caManager->getTypeId(_typeName);
	chemData.concField = caManager->getConcentrationField(_fieldName);
	chemData.lambda=_lambda;

	vecChemotaxisDataByType[chemData.type].push_back(chemData);

}
//////////////////////////////////////////////////////////////////////////////////////////
void ChemotaxisProbability::clearChemotaxisData(){
	vecChemotaxisDataByType.assign(UCHAR_MAX+1,vector<ChemotaxisData>());
}

//////////////////////////////////////////////////////////////////////////////////////////
std::string ChemotaxisProbability::toString(){return "ChemotaxisProbability";}


float ChemotaxisProbability::calculate(const CACell * _sourceCell, const Point3D & _source, const Point3D & _target){
	
	
	if (!_sourceCell) return 0.0;

	vector<ChemotaxisData> & chemotaxisDataVec = vecChemotaxisDataByType[_sourceCell->type];
	if (! chemotaxisDataVec .size()) return 0.0;

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

	float prob = 0.0;
	for (int i  = 0 ; i < chemotaxisDataVec .size() ; ++i){
		ChemotaxisData & chemData = chemotaxisDataVec[i];
		cerr<<"_source="<<_source<<" _target="<<_target<<endl;
		cerr<<"celltype="<<(int)_sourceCell->type<<endl;
		cerr<<"chemData.concField->get(_target)="<<chemData.concField->get(_target)<<" chemData.concField->get(_source)="<<chemData.concField->get(_source)<<endl;
		cerr<<"chemData.lambda="<<chemData.lambda<<endl;
		cerr<<"deltaT="<<deltaT<<endl;
		cerr<<"_source="<<_source<<" _target="<<_target<<endl;
		cerr<<"targetFillLevel ="<<targetFillLevel <<endl;
		cerr<<"carryingCapacity  = " <<carryingCapacity <<endl;

		prob += (chemData.lambda/2.0)*deltaT/(2*((_source.x-_target.x)*(_source.x-_target.x)+(_source.y-_target.y)*(_source.y-_target.y)+(_source.z-_target.z)*(_source.z-_target.z)))	
		*(chemData.concField->get(_target)-chemData.concField->get(_source))
	    *(carryingCapacity-targetFillLevel)/(float)(carryingCapacity);
	}
	//float prob = diffCoeff*deltaT/(2*((_source.x-_target.x)*(_source.x-_target.x)+(_source.y-_target.y)*(_source.y-_target.y)+(_source.z-_target.z)*(_source.z-_target.z)))
	//*(carryingCapacity-targetFillLevel)/(float)(carryingCapacity);

	if (targetStack ){
		
#ifdef _DEBUG
		cerr<<"prob="<<prob<<endl;
#endif
	}
	cerr<<"prob="<<prob<<endl;
    return prob;
}    