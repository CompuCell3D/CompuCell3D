


#include <iostream>
#include <string>
#include <Components/CellCM.h>
#include <Components/SimulationBox.h>
#include <Components/CellInventoryCM.h>
#include <Components/CellFactoryCM.h>
#include <Components/SimulatorCM.h>
#include <Components/IntegratorFE.h>

#include <time.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <PublicUtilities/NumericalUtils.h>
#include <limits>
#include <fstream>

#include <BasicUtils/BasicModuleManager.h>
#include <Components/Interfaces/ForceTerm.h>


#include <Components/Interfaces/ForceTerm.h>

#include <XMLUtils/XMLParserExpat.h>



#if defined(_WIN32)
	#include <windows.h>
#endif


using namespace std;
using namespace  CenterModel;



void Syntax(const string name) {
  cerr << "Syntax: " << name << " <config>" << endl;
  exit(1);
}




int main(int argc, char *argv[]) {

	cerr<<"THE NEW Welcome to CC3D command line edition"<<endl;
    CellCM cell;
    cell.grow();
	cerr<<"cell.position="<<cell.position<<endl;

    //BasicModuleManager<ForceTerm> forceTermManager;
    

   XMLParserExpat xmlParser;
	if(argc<2){
        cerr<<"argc="<<argc<<endl;    
		cerr<<"SPECIFY XML FILE"<<endl;
		exit(0);
	}
    xmlParser.setFileName(string(argv[1]));
	xmlParser.parse();

    CC3DXMLElement *simulatorData=xmlParser.rootElement->getFirstElement("Simulator");
    cerr<<"simulatorData="<<simulatorData<<endl;

	//extracting force term elements from the XML file
	CC3DXMLElementList forceTermDataList=xmlParser.rootElement->getElements("ForceTerm");
    cerr<<"forceTermDataList.size()="<<forceTermDataList.size()<<endl;
	for(int i = 0 ; i < forceTermDataList.size() ; ++i ){
		cerr<<"THIS IS ForceTerm: "<<forceTermDataList[i]->getAttribute("Name")<<endl;
        //sim.ps.addPluginDataCC3D(pluginDataList[i]);        
	}

    

    SimulatorCM simulator;
    

    SimulatorCM::forceTermManager_t * forceTermManagerPtr=simulator.getForceTermManagerPtr();

    char *forceTermPath = getenv("COMPUCELL3D_FORCECM_PATH");
    cerr<<"forceTermPath ="<<forceTermPath <<endl;
    if (forceTermPath ) forceTermManagerPtr->scanLibraries(forceTermPath);

    //ForceTerm *ljTm=0;
    //ljTm = forceTermManagerPtr->get("LennardJones");

    //if (!ljTm){
    //    cerr<<"ljTm="<<ljTm<<" COULD NOT FIND REQUESTED MODULE"<<endl;
    //    return 0;
    //}

    //cerr<<"ljTm="<<ljTm<<endl;
    //

    //ForceTerm *stochForce=0;
    //stochForce=forceTermManagerPtr->get("Stochastic");
    //cerr<<"StochForce="<<stochForce<<endl;


    simulator.setBoxDim(21.2,45.7,80.1);
    simulator.setGridSpacing(2.01,2.01,2.01);
    simulator.setBoundaryConditionVec(0.,0.,0.); //no flux bc

        
    simulator.init();

    
	//loading force terms
	for(int i = 0 ; i < forceTermDataList.size() ; ++i ){
		cerr<<"THIS IS ForceTerm: "<<forceTermDataList[i]->getAttribute("Name")<<endl;
        simulator.handleForceTermRequest(forceTermDataList[i]);
        //sim.ps.addPluginDataCC3D(pluginDataList[i]);        
	}

    int N=20000;
    double r_min=1.0;
    double r_max=2.0;
    double mot_min=30000.0;
    double mot_max=60000.0;

    simulator.createRandomCells(N,r_min,r_max,mot_min,mot_max);
    ////////CellCM * cellTmp;

    ////////CellInventoryCM * ciPtr=simulator.getCellInventoryPtr();
    ////////CellFactoryCM * cfPtr = simulator.getCellFactoryPtr();

    ////////cellTmp=cfPtr->createCellCM(11.02,35.1,51.7);
    ////////cellTmp->interactionRadius=1.6;
    ////////cellTmp->effectiveMotility=20000.0;
    ////////ciPtr->addToInventory(cellTmp);

    ////////cellTmp=cfPtr->createCellCM(11.02,35.7,51.7);
    ////////cellTmp->interactionRadius=1.6;
    ////////cellTmp->effectiveMotility=20000.0;
    ////////ciPtr->addToInventory(cellTmp);


    ////////cellTmp=cfPtr->createCellCM(11.1,35.7,51.7);
    ////////cellTmp->interactionRadius=1.6;
    ////////cellTmp->effectiveMotility=20000.0;
    ////////ciPtr->addToInventory(cellTmp);

    

    //ljTm->init(&simulator);   
    //stochForce->init(&simulator);   
    //cerr<<"after ljTm->init(&simulator)"<<endl;

    //simulator.registerForce(ljTm);
    
    //simulator.registerForce(ljTm);
    //simulator.registerForce(ljTm);
    //simulator.registerForce(ljTm);
    //simulator.registerForce(ljTm);
    //simulator.registerForce(ljTm);
    //simulator.registerForce(ljTm);
    //simulator.registerForce(ljTm);
    //simulator.registerForce(ljTm);
    //simulator.registerForce(ljTm);


    //LennardJonesForceTerm ljTerm;

    //ljTerm.init(&simulator);

    //simulator.registerForce(&ljTerm);
    //simulator.registerForce(&ljTerm);
    //simulator.registerForce(&ljTerm);
    //simulator.registerForce(&ljTerm);
    //simulator.registerForce(&ljTerm);
    //simulator.registerForce(&ljTerm);
    //simulator.registerForce(&ljTerm);
    //simulator.registerForce(&ljTerm);
    //simulator.registerForce(&ljTerm);
    //simulator.registerForce(&ljTerm);


    IntegratorFE integrator;
    IntegratorData * integrDataPtr=integrator.getIntegratorDataPtr();
    integrDataPtr->tolerance=0.1;
    simulator.registerIntegrator(&integrator);

    double endTime=100.0;

#if defined(_WIN32)
	volatile DWORD dwStart;
	dwStart = GetTickCount();
#endif
    while (simulator.getCurrentTime()<endTime){

        simulator.step();

    }



#if defined(_WIN32)
	cerr<<"DISTANCE CALCULATION FOR  "<<N<<" cells too "<<GetTickCount()-dwStart<<" miliseconds to complete"<<endl;
	dwStart = GetTickCount();
#endif	



	
	
  return 1;
}
