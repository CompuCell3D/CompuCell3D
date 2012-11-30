#include "ForceCalculator.h"
#include "SimulationBox.h"
#include "CellInventoryCM.h"
#include <Components/Interfaces/ForceTerm.h>
#include "SimulatorCM.h"
#include <PublicUtilities/NumericalUtils.h>

using namespace CenterModel;

ForceCalculator::ForceCalculator():sbPtr(0),ciPtr(0),simulator(0){}


ForceCalculator::~ForceCalculator(){}


void ForceCalculator::init(SimulatorCM *_simulator){
    simulator=_simulator;
    sbPtr = simulator->getSimulationBoxPtr();
    ciPtr = simulator->getCellInventoryPtr();    
}

void ForceCalculator::registerForce(ForceTerm * _forceTerm){
    if (! _forceTerm)
        return;

    forceTermRegistry.push_back(_forceTerm);
}


void ForceCalculator::calculateForces(){

    CellInventoryCM &ci=*ciPtr;
    SimulationBox &sb=*sbPtr;
    Vector3 boxDim=simulator->getBoxDim();
    Vector3 bc=simulator->getBoundaryConditionVec();

	int n=0;
	double dist=0.0;

    double potential;
    double A=0.2, B=0.1;
    double forceMag;
    Vector3 distVec;
    Vector3 unitDistVec;
    Vector3 netForce;

	CellInventoryCM::cellInventoryIterator itr;
	for (itr=ci.cellInventoryBegin() ; itr!=ci.cellInventoryEnd(); ++itr){
		CellCM * cell=itr->second;
        netForce=Vector3(0.,0.,0.);
        cell->netForce=Vector3(0.,0.,0.);


        InteractionRangeIterator itr = sb.getInteractionRangeIterator(cell).begin();

		//itr.begin();
		InteractionRangeIterator endItr = sb.getInteractionRangeIterator(cell).end();

		//cerr<<"***************** NEIGHBORS of cell->id="<<cell->id<<endl;
		CellCM *nCell;
		for (itr.begin(); itr!=endItr ;++itr){
			nCell=(*itr);
			if (nCell==cell)//neglecting "self interactions"
				continue; 

            distVec=distanceVectorInvariantCenterModel(cell->position,nCell->position,boxDim ,bc);
            dist=distVec.Mag();
            unitDistVec=distVec.Unit();
            //cerr<<"forceTermRegistry.size()="<<forceTermRegistry.size()<<endl;

            if (dist<=cell->interactionRadius || dist<=nCell->interactionRadius){

                for (int i = 0 ; i < forceTermRegistry.size() ; ++i){
                    netForce+=forceTermRegistry[i]->forceTerm(cell,nCell,dist,unitDistVec);
                    //cerr<<"forceTerm["<<i<<"]="<<netForce<<endl;
                }
            
                cell->netForce=netForce;
            }            
			
		}
		


	}


}


