
#include "SimulatorCM.h"
#include "CellCM.h"

#include <time.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <PublicUtilities/NumericalUtils.h>
#include <limits>
#include <stdlib.h>

#if defined(_WIN32)
	#include <windows.h>
#endif
		
using namespace CenterModel;

SimulatorCM::SimulatorCM():stepCounter(0),timeSim(0.0)

{}

SimulatorCM::~SimulatorCM(){}


void SimulatorCM::init(){
    sb.setBoxSpatialProperties(boxDim,gridSpacing);
    cf.setSimulationBox(&sb);
    ci.setCellFactory(&cf);
    fCalc.init(this);
}


void SimulatorCM::registerForce(ForceTerm * _forceTerm){
    fCalc.registerForce(_forceTerm);
}

void SimulatorCM::createRandomCells(int N, double r_min, double r_max){

	BasicRandomNumberGeneratorNonStatic rGen;
    //srand(GetTickCount());
	srand((unsigned)time(0));
    //srand(time( NULL ));
	unsigned int randomSeed=(unsigned int)rand()*((std::numeric_limits<unsigned int>::max)()-1);                
	rGen.setSeed(randomSeed);

	
	for (int i=0;i<N;++i){
		CellCM * cell=cf.createCellCM(boxDim.fX*rGen.getRatio(),boxDim.fY*rGen.getRatio(),boxDim.fZ*rGen.getRatio());
		cell->interactionRadius=r_min+rGen.getRatio()*(r_max-r_min);
		ci.addToInventory(cell);
	}

}

void SimulatorCM::step(){
    fCalc.calculateForces();

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
		
