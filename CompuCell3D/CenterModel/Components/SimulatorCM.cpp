
#include "SimulatorCM.h"
#include "CellCM.h"
#include "Integrator.h"
#include <XMLUtils/CC3DXMLElement.h>
#include <time.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <PublicUtilities/NumericalUtils.h>
#include <limits>
#include <stdlib.h>
#include <Components/Interfaces/ForceTerm.h>

#if defined(_WIN32)
	#include <windows.h>
#endif
		
using namespace CenterModel;
using namespace std;

SimulatorCM::SimulatorCM():stepCounter(0),timeSim(0.0),integrator(0)

{}

SimulatorCM::~SimulatorCM(){}


void SimulatorCM::init(){
    sb.setBoxSpatialProperties(boxDim,gridSpacing);
    cf.setSimulationBox(&sb);
    ci.setCellFactory(&cf);
    fCalc.init(this);
}

void SimulatorCM::handleForceTermRequest(CC3DXMLElement * _forceElement){

    std::string moduleName=_forceElement->getAttribute("Name");

    bool moduleAlreadyRegisteredFlag=false;

    //ForceTerm * module=forceTermManager.get(moduleName,&moduleAlreadyRegisteredFlag);
	CenterModelObject * module=(CenterModelObject *)forceTermManager.get(moduleName,&moduleAlreadyRegisteredFlag);

	
	if(!moduleAlreadyRegisteredFlag){
		//Will only process first occurence of a given plugin
		cerr<<"INITIALIZING "<<moduleName<<endl;
		module->init(this, _forceElement);
    }else{
        cerr<<" MODULE "<<moduleName<<" is laready registered - ignoring request"<<endl;
    }


}


void SimulatorCM::registerForce(ForceTerm * _forceTerm){
    fCalc.registerForce(_forceTerm);
}

void SimulatorCM::registerSingleBodyForce(SingleBodyForceTerm * _forceTerm){
    fCalc.registerSingleBodyForce(_forceTerm);
}


void SimulatorCM::registerIntegrator(Integrator * _integrator){
    integrator=_integrator;
    integrator->init(this);
}

void SimulatorCM::createRandomCells(int N, double r_min, double r_max, double mot_min, double mot_max){

	BasicRandomNumberGeneratorNonStatic rGen;
    //srand(GetTickCount());
	srand((unsigned)time(0));
    //srand(time( NULL ));
	unsigned int randomSeed=(unsigned int)rand()*((std::numeric_limits<unsigned int>::max)()-1);                
	rGen.setSeed(randomSeed);

	
	for (int i=0;i<N;++i){
		CellCM * cell=cf.createCellCM(boxDim.fX*rGen.getRatio(),boxDim.fY*rGen.getRatio(),boxDim.fZ*rGen.getRatio());
		cell->interactionRadius=r_min+rGen.getRatio()*(r_max-r_min);
        cell->effectiveMotility=mot_min+rGen.getRatio()*(mot_max-mot_min);
		ci.addToInventory(cell);
	}

}

void SimulatorCM::step(){
    //cerr<<"SIMULATOR STEP="<<endl;

    fCalc.calculateForces();
    //cerr<<"AFTER calculateForces="<<endl;
    IntegratorData * integrDataPtr=integrator->getIntegratorDataPtr();    
    

    //cerr<<"BEFORE integrate"<<endl;
    integrator->integrate();

    timeSim+=integrator->getIntegratorDataPtr()->dt;
    ++stepCounter;
    cerr<<"stepCounter="<<stepCounter<<" time="<<timeSim<<" time_step="<<integrator->getIntegratorDataPtr()->dt<<endl;
    //cerr<<"AFTER integrate"<<endl;

    return;
	//int n=0;
	//double dist=0.0;

 //   double potential;
 //   double A=0.2, B=0.1;
 //   double forceMag;
 //   Vector3 distVector;

	//CellInventoryCM::cellInventoryIterator itr;
	//for (itr=ci.cellInventoryBegin() ; itr!=ci.cellInventoryEnd(); ++itr){
	//	CellCM * cell=itr->second;
 //       cell->netForce=Vector3(0.,0.,0.);

	//	InteractionRangeIterator itr = sb.getInteractionRangeIterator(cell);

	//	itr.begin();
	//	InteractionRangeIterator endItr = sb.getInteractionRangeIterator(cell).end();

	//	//cerr<<"***************** NEIGHBORS of cell->id="<<cell->id<<endl;
	//	CellCM *nCell;
	//	for (itr.begin(); itr!=endItr ;++itr){
	//		nCell=(*itr);
	//		if (nCell==cell)//neglecting "self interactions"
	//			continue; 
 //           distVector=distanceVectorInvariantCenterModel(cell->position,nCell->position,boxDim ,bc);
 //           dist=distVector.Mag();
 //           potential=A*pow(dist,-12.0)-B*pow(dist,-6.0);
 //           forceMag=A*(-12.0)*pow(dist,-13.0)-B*(-6.0)*pow(dist,-7.0);
 //           cell->netForce+=distVector.Unit()*forceMag;
 //           
 //           //potential=0.1*pow(dist,6)-0.2*pow(dist,12);
	//		if (dist<=cell->interactionRadius || dist<=nCell->interactionRadius){

	//			//cerr<<"**********INTERACTION "<<cell->id<<"-"<<nCell->id<<" ********************"<<endl;
	//			//cerr<<"THIS IS cell.id="<<nCell->id<<" distance from center cell="<<dist<<endl;
	//			//cerr<<"nCell->position="<<nCell->position<<" cell->position="<<cell->position<<endl;
 //   //            cerr<<"nCellLocation="<<sb.getCellLatticeLocation(nCell)<<" cell location="<<sb.getCellLatticeLocation(cell)<<endl;
	//			++n;
	//		}

	//		
	//	}
	//	
	//	//if (n++>20){
	//	//	break;
	//	//}
	//


	//}


}
		
