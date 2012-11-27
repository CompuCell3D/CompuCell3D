#include "IntegratorFE.h"
#include "CellInventoryCM.h"
#include "SimulationBox.h"
#include <iostream>

using namespace std;

using namespace CenterModel;

IntegratorFE::IntegratorFE(){}
        
IntegratorFE::~IntegratorFE(){}

void IntegratorFE::init(SimulatorCM *_simulator){
    Integrator::init(_simulator);
}

double IntegratorFE::calculateTimeStep(){
    CellInventoryCM::cellInventoryIterator itr;    

    double maxAbsVelocity=0.0;
    double absVelocity;

	for (itr=ciPtr->cellInventoryBegin() ; itr!=ciPtr->cellInventoryEnd(); ++itr){
		CellCM * cell=itr->second;
        absVelocity=fabs(cell->netForce.Mag()/cell->effectiveMotility);
        if (absVelocity>maxAbsVelocity){
            maxAbsVelocity=absVelocity;
        }
    }

    
    
    integratorData.dt=integratorData.tolerance/maxAbsVelocity;

    return integratorData.dt;

}

void IntegratorFE::integrate(){

    calculateTimeStep();

    CellInventoryCM::cellInventoryIterator itr;    

	for (itr=ciPtr->cellInventoryBegin() ; itr!=ciPtr->cellInventoryEnd(); ++itr){
		CellCM * cell=itr->second;

        //cerr<<"cellid="<<cell->id<<"cell->netForce="<<cell->netForce<<endl;
        Vector3 positionBefore=cell->position;
        cell->velocity=(1.0/cell->effectiveMotility)*cell->netForce;

        cell->position+=cell->velocity*integratorData.dt;

        long int lookupIdx=cell->lookupIdx;
        //cerr<<"integratorData.dt="<<integratorData.dt<<endl;
        //cerr<<"cell->id="<<cell->id<<" pos="<<cell->position<<endl;
        //cerr<<"cell->netForce="<<cell->netForce<<endl;
        //cerr<<"cell->effectiveMotility="<<cell->effectiveMotility<<endl;
        //cerr<<"cell->velocity*integratorData.dt="<<cell->velocity*integratorData.dt<<endl;

        sbPtr->updateCellLookup(cell); //updating lookup information about cell

        if (lookupIdx!=cell->lookupIdx){
            cerr<<"cell->id="<<cell->id<<" idx before="<<lookupIdx<<" index after="<<cell->lookupIdx<<endl;
            cerr<<"positionBefore="<<positionBefore<<" position after="<<cell->position<<endl;
            
        }
    }

}		

