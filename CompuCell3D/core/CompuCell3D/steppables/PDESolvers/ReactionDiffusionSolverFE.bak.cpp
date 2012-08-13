

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3DIO.h>
#include <BasicUtils/BasicClassGroup.h>
#include <CompuCell3D/steppables/BoxWatcher/BoxWatcher.h>

#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <PublicUtilities/StringUtils.h>
#include <muParser/muParser.h>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <PublicUtilities/ParalellUtilsOpenMP.h>


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// std::ostream & operator<<(std::ostream & out,CompuCell3D::DiffusionData & diffData){
//
//
// }


using namespace CompuCell3D;
using namespace std;


#include "ReactionDiffusionSolverFE.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverSerializer::serialize(){

	for(int i = 0 ; i < solverPtr->diffSecrFieldTuppleVec.size() ; ++i){
		ostringstream outName;

		outName<<solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName<<"_"<<currentStep<<"."<<serializedFileExtension;
		ofstream outStream(outName.str().c_str());
		solverPtr->outputField( outStream,solverPtr->concentrationFieldVector[i]);
	}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverSerializer::readFromFile(){
	try{
		for(int i = 0 ; i < solverPtr->diffSecrFieldTuppleVec.size() ; ++i){
			ostringstream inName;
			inName<<solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName<<"."<<serializedFileExtension;

			solverPtr->readConcentrationField(inName.str().c_str(),solverPtr->concentrationFieldVector[i]);;
		}

	} catch (BasicException &e) {
		cerr<<"COULD NOT FIND ONE OF THE FILES"<<endl;
		throw BasicException("Error in reading diffusion fields from file",e);
	}


}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ReactionDiffusionSolverFE::ReactionDiffusionSolverFE()
: DiffusableVectorContiguous<float>(),deltaX(1.0),deltaT(1.0)
{
	serializerPtr=0;
	serializeFlag=false;
	pUtils=0;
	readFromFileFlag=false;
	haveCouplingTerms=false;
	serializeFrequency=0;
	boxWatcherSteppable=0;
	cellTypeVariableName="CellType";
	useBoxWatcher=false;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ReactionDiffusionSolverFE::~ReactionDiffusionSolverFE()
{
	if(serializerPtr)
		delete serializerPtr ; serializerPtr=0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {


	simPtr=_simulator;
	simulator=_simulator;
	potts = _simulator->getPotts();
	automaton=potts->getAutomaton();

	///getting cell inventory
	cellInventoryPtr=& potts->getCellInventory();

	///getting field ptr from Potts3D
	///**
	//   cellFieldG=potts->getCellFieldG();
	cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	fieldDim=cellFieldG->getDim();

	pUtils=simulator->getParallelUtils();

	cerr<<"INSIDE INIT"<<endl;



	///setting member function pointers
	diffusePtr=&ReactionDiffusionSolverFE::diffuse;
	secretePtr=&ReactionDiffusionSolverFE::secrete;


	update(_xmlData,true);

	//numberOfFields=diffSecrFieldTuppleVec.size();




	vector<string> concentrationFieldNameVectorTmp; //temporary vector for field names
	///assign vector of field names
	concentrationFieldNameVectorTmp.assign(diffSecrFieldTuppleVec.size(),string(""));

	cerr<<"diffSecrFieldTuppleVec.size()="<<diffSecrFieldTuppleVec.size()<<endl;

	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		//       cerr<<" concentrationFieldNameVector[i]="<<diffDataVec[i].fieldName<<endl;
		//       concentrationFieldNameVector.push_back(diffDataVec[i].fieldName);
		concentrationFieldNameVectorTmp[i] = diffSecrFieldTuppleVec[i].diffData.fieldName;
		cerr<<" concentrationFieldNameVector[i]="<<concentrationFieldNameVectorTmp[i]<<endl;
	}

	////setting up couplingData - field-field interaction terms
	//vector<CouplingData>::iterator pos;

	//for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
	//	pos=diffSecrFieldTuppleVec[i].diffData.couplingDataVec.begin();
	//	for(int j = 0 ; j < diffSecrFieldTuppleVec[i].diffData.couplingDataVec.size() ; ++j){

	//		for(int idx=0; idx<concentrationFieldNameVectorTmp.size() ; ++idx){
	//			if( concentrationFieldNameVectorTmp[idx] == diffSecrFieldTuppleVec[i].diffData.couplingDataVec[j].intrFieldName ){
	//				diffSecrFieldTuppleVec[i].diffData.couplingDataVec[j].fieldIdx=idx;
	//				haveCouplingTerms=true; //if this is called at list once we have already coupling terms and need to proceed differently with scratch field initialization
	//				break;
	//			}
	//			//this means that required interacting field name has not been found
	//			if( idx == concentrationFieldNameVectorTmp.size()-1 ){
	//				//remove this interacting term
	//				//                pos=&(diffDataVec[i].degradationDataVec[j]);
	//				diffSecrFieldTuppleVec[i].diffData.couplingDataVec.erase(pos);
	//			}
	//		}
	//		++pos;
	//	}
	//}


	cerr<<"FIELDS THAT I HAVE"<<endl;
	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		cerr<<"Field "<<i<<" name: "<<concentrationFieldNameVectorTmp[i]<<endl;
	}




	///allocate fields including scrartch field
	allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size(),fieldDim); 
	workFieldDim=concentrationFieldVector[0]->getInternalDim();

	//workFieldDim=Dim3D(fieldDim.x+2,fieldDim.y+2,fieldDim.z+2);
	/////allocate fields including scrartch field
	//allocateDiffusableFieldVector(2*diffSecrFieldTuppleVec.size(),workFieldDim); //with coupling terms every field need to have its own scratch field

	//here I need to copy field names from concentrationFieldNameVectorTmp to concentrationFieldNameVector
	//because concentrationFieldNameVector is reallocated with default values once I call allocateDiffusableFieldVector


	for(unsigned int i=0 ; i < concentrationFieldNameVectorTmp.size() ; ++i){
		concentrationFieldNameVector[i]=concentrationFieldNameVectorTmp[i];
	}


	//register fields once they have been allocated
	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		simPtr->registerConcentrationField(concentrationFieldNameVector[i] , concentrationFieldVector[i]);
		cerr<<"registring field: "<<concentrationFieldNameVector[i]<<" field address="<<concentrationFieldVector[i]<<endl;
	}





	//    exit(0);

	periodicBoundaryCheckVector.assign(3,false);
	string boundaryName;
	boundaryName=potts->getBoundaryXName();
	changeToLower(boundaryName);
	if(boundaryName=="periodic")  {
		periodicBoundaryCheckVector[0]=true;
	}
	boundaryName=potts->getBoundaryYName();
	changeToLower(boundaryName);
	if(boundaryName=="periodic")  {
		periodicBoundaryCheckVector[1]=true;
	}

	boundaryName=potts->getBoundaryZName();
	changeToLower(boundaryName);
	if(boundaryName=="periodic")  {
		periodicBoundaryCheckVector[2]=true;
	}

	simulator->registerSteerableObject(this);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::extraInit(Simulator *simulator){

	if((serializeFlag || readFromFileFlag) && !serializerPtr){
		serializerPtr=new ReactionDiffusionSolverSerializer();
		serializerPtr->solverPtr=this;
	}

	if(serializeFlag){
		simulator->registerSerializer(serializerPtr);
	}

	//bool useBoxWatcher=false;
	//for (int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
	//	if(diffSecrFieldTuppleVec[i].diffData.useBoxWatcher){
	//		useBoxWatcher=true;
	//		break;
	//	}
	//}
	bool steppableAlreadyRegisteredFlag;
	if(useBoxWatcher){
		boxWatcherSteppable=(BoxWatcher*)Simulator::steppableManager.get("BoxWatcher",&steppableAlreadyRegisteredFlag);
		if(!steppableAlreadyRegisteredFlag)
			boxWatcherSteppable->init(simulator);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::start() {
	//     if(diffConst> (1.0/6.0-0.05) ){ //hard coded condtion for stability of the solutions - assume dt=1 dx=dy=dz=1
	//
	//       cerr<<"CANNOT SOLVE DIFFUSION EQUATION: STABILITY PROBLEM - DIFFUSION CONSTANT TOO LARGE. EXITING..."<<endl;
	//       exit(0);
	//
	//    }


	dt_dx2=deltaT/(deltaX*deltaX);
	if(readFromFileFlag){
		try{

			serializerPtr->readFromFile();

		} catch (BasicException &e){
			cerr<<"Going to fail-safe initialization"<<endl;
			initializeConcentration(); //if there was error, initialize using failsafe defaults
		}

	}else{
		initializeConcentration();//Normal reading from User specified files
	}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ReactionDiffusionSolverFE::initializeConcentration()
{
	for(unsigned int i = 0 ; i <diffSecrFieldTuppleVec.size() ; ++i)
	{
		if(diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;
		cerr << "fail-safe initialization " << diffSecrFieldTuppleVec[i].diffData.concentrationFileName << endl;
		readConcentrationField(diffSecrFieldTuppleVec[i].diffData.concentrationFileName,concentrationFieldVector[i]);
	}


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::step(const unsigned int _currentStep) {

	currentStep=_currentStep;

	(this->*secretePtr)();

	(this->*diffusePtr)();


	if(serializeFrequency>0 && serializeFlag && !(_currentStep % serializeFrequency)){
		serializerPtr->setCurrentStep(currentStep);
		serializerPtr->serialize();
	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ReactionDiffusionSolverFE::secreteOnContactSingleField(unsigned int idx){


	SecretionData & secrData=diffSecrFieldTuppleVec[idx].secrData;


	std::map<unsigned char,SecretionOnContactData>::iterator mitr;
	std::map<unsigned char,SecretionOnContactData>::iterator end_mitr=secrData.typeIdSecrOnContactDataMap.end();

	CellG *currentCellPtr;


	ConcentrationField_t & concentrationField=*concentrationFieldVector[idx];
	//Array3D_t & concentrationArray = concentrationFieldPtr->getContainer();

	float currentConcentration;
	float secrConst;
	float secrConstMedium=0.0;
	std::map<unsigned char, float> * contactCellMapMediumPtr;
	std::map<unsigned char, float> * contactCellMapPtr;
	std::map<unsigned char, float>::iterator mitrTypeConst;

	bool secreteInMedium=false;
	//the assumption is that medium has type ID 0
	mitr=secrData.typeIdSecrOnContactDataMap.find(automaton->getTypeId("Medium"));

	if( mitr != end_mitr ){
		secreteInMedium=true;
		contactCellMapMediumPtr = &(mitr->second.contactCellMap);
	}



	Point3D pt;
	Neighbor n;
	CellG *nCell=0;
	WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
	unsigned char type;



	unsigned x_min=1,x_max=fieldDim.x+1;
	unsigned y_min=1,y_max=fieldDim.y+1;
	unsigned z_min=1,z_max=fieldDim.z+1;

	for (int z = z_min; z < z_max; z++)
		for (int y = y_min; y < y_max; y++)
			for (int x = x_min; x < x_max; x++){
				pt=Point3D(x-1,y-1,z-1);
				///**
				currentCellPtr=cellFieldG->get(pt);
				//             currentCellPtr=cellFieldG->get(pt);
				currentConcentration = concentrationField.getDirect(x,y,z);



				if(secreteInMedium && ! currentCellPtr){

					for (int i = 0  ; i<=maxNeighborIndex/*offsetVec.size()*/ ; ++i ){
						n=boundaryStrategy->getNeighborDirect(pt,i);
						if(!n.distance)//not a valid neighbor
							continue;
						///**
						nCell = fieldG->get(n.pt);
						//                      nCell = fieldG->get(n.pt);
						if(nCell)
							type=nCell->type;
						else
							type=0;


						mitrTypeConst=contactCellMapMediumPtr->find(type);

						if(mitrTypeConst != contactCellMapMediumPtr->end()){//OK to secrete, contact detected
							secrConstMedium = mitrTypeConst->second;

							concentrationField.setDirect(x,y,z,currentConcentration+secrConstMedium);
						}



					}

					continue;
				}

				if(currentCellPtr){
					mitr=secrData.typeIdSecrOnContactDataMap.find(currentCellPtr->type);
					if(mitr!=end_mitr){

						contactCellMapPtr = &(mitr->second.contactCellMap);

						for (int i = 0  ; i<=maxNeighborIndex/*offsetVec.size() */; ++i ){

							n=boundaryStrategy->getNeighborDirect(pt,i);
							if(!n.distance)//not a valid neighbor
								continue;
							///**
							nCell = fieldG->get(n.pt);
							//                      nCell = fieldG->get(n.pt);
							if(nCell)
								type=nCell->type;
							else
								type=0;

							if (currentCellPtr==nCell) continue; //skip secretion in pixels belongin to the same cell

							mitrTypeConst=contactCellMapPtr->find(type);
							if(mitrTypeConst != contactCellMapPtr->end()){//OK to secrete, contact detected
								secrConst=mitrTypeConst->second;
								//                         concentrationField->set(pt,currentConcentration+secrConst);
								concentrationField.setDirect(x,y,z,currentConcentration+secrConst);
							}

						}


					}
				}
			}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::secreteSingleField(unsigned int idx){


	SecretionData & secrData=diffSecrFieldTuppleVec[idx].secrData;


	std::map<unsigned char,float>::iterator mitr;	
	std::map<unsigned char,float>::iterator end_mitr=secrData.typeIdSecrConstMap.end();
	std::map<unsigned char,UptakeData>::iterator mitrUptake;
	std::map<unsigned char,UptakeData>::iterator end_mitrUptake=secrData.typeIdUptakeDataMap.end();



	CellG *currentCellPtr;
	//Field3DImpl<float> * concentrationField=concentrationFieldVector[idx];
	float currentConcentration;
	float secrConst;
	float secrConstMedium=0.0;
	float maxUptakeInMedium=0.0;
	float relativeUptakeRateInMedium=0.0;

	ConcentrationField_t & concentrationField=*concentrationFieldVector[idx];
	//Array3D_t & concentrationArray = concentrationFieldPtr->getContainer();

	bool doUptakeFlag=false;
	bool uptakeInMediumFlag=false;
	bool secreteInMedium=false;
	//the assumption is that medium has type ID 0
	mitr=secrData.typeIdSecrConstMap.find(automaton->getTypeId("Medium"));

	if( mitr != end_mitr){
		secreteInMedium=true;
		secrConstMedium=mitr->second;
	}

	//uptake for medium setup
	if(secrData.typeIdUptakeDataMap.size()){
		doUptakeFlag=true;
	}
	//uptake for medium setup
	if(doUptakeFlag){
		mitrUptake=secrData.typeIdUptakeDataMap.find(automaton->getTypeId("Medium"));
		if(mitrUptake != end_mitrUptake){
			maxUptakeInMedium=mitrUptake->second.maxUptake;
			relativeUptakeRateInMedium=mitrUptake->second.relativeUptakeRate;
			uptakeInMediumFlag=true;

		}
	}


	Point3D pt;

	unsigned x_min=1,x_max=fieldDim.x+1;
	unsigned y_min=1,y_max=fieldDim.y+1;
	unsigned z_min=1,z_max=fieldDim.z+1;

	for (int z = 1; z < workFieldDim.z-1; z++)
		for (int y = 1; y < workFieldDim.y-1; y++)
			for (int x = 1; x < workFieldDim.x-1; x++){
				pt=Point3D(x-1,y-1,z-1);
				//             cerr<<"pt="<<pt<<" is valid "<<cellFieldG->isValid(pt)<<endl;
				///**
				currentCellPtr=cellFieldG->get(pt);
				//             currentCellPtr=cellFieldG->get(pt);
				//             cerr<<"THIS IS PTR="<<currentCellPtr<<endl;

				//             if(currentCellPtr)
				//                cerr<<"This is id="<<currentCellPtr->id<<endl;
				currentConcentration = concentrationField.getDirect(x,y,z);

				if(secreteInMedium && ! currentCellPtr){
					concentrationField.setDirect(x,y,z,currentConcentration+secrConstMedium);
				}

				if(currentCellPtr){
					mitr=secrData.typeIdSecrConstMap.find(currentCellPtr->type);
					if(mitr!=end_mitr){
						secrConst=mitr->second;
						concentrationField.setDirect(x,y,z,currentConcentration+secrConst);
					}
				}

				if(doUptakeFlag){
					if(uptakeInMediumFlag && ! currentCellPtr){
						if(currentConcentration>maxUptakeInMedium){
							concentrationField.setDirect(x,y,z,currentConcentration-maxUptakeInMedium);	
						}else{
							concentrationField.setDirect(x,y,z,currentConcentration - currentConcentration*relativeUptakeRateInMedium);	
						}
					}
					if(currentCellPtr){

						mitrUptake=secrData.typeIdUptakeDataMap.find(currentCellPtr->type);
						if(mitrUptake!=end_mitrUptake){
							if(currentConcentration > mitrUptake->second.maxUptake){
								concentrationField.setDirect(x,y,z,currentConcentration-mitrUptake->second.maxUptake);
								//cerr<<" uptake concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<mitrUptake->second.maxUptake<<endl;
							}else{
								//cerr<<"concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<currentConcentration*mitrUptake->second.relativeUptakeRate<<endl;
								concentrationField.setDirect(x,y,z,currentConcentration- currentConcentration*mitrUptake->second.relativeUptakeRate);	
							}
						}
					}
				}
			}


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::secreteConstantConcentrationSingleField(unsigned int idx){


	SecretionData & secrData=diffSecrFieldTuppleVec[idx].secrData;


	std::map<unsigned char,float>::iterator mitr;
	std::map<unsigned char,float>::iterator end_mitr=secrData.typeIdSecrConstConstantConcentrationMap.end();

	CellG *currentCellPtr;
	//Field3DImpl<float> * concentrationField=concentrationFieldVector[idx];
	float currentConcentration;
	float secrConst;
	float secrConstMedium=0.0;

	ConcentrationField_t & concentrationField=*concentrationFieldVector[idx];
	//Array3D_t & concentrationArray = concentrationFieldPtr->getContainer();


	bool secreteInMedium=false;
	//the assumption is that medium has type ID 0
	mitr=secrData.typeIdSecrConstConstantConcentrationMap.find(automaton->getTypeId("Medium"));

	if( mitr != end_mitr){
		secreteInMedium=true;
		secrConstMedium=mitr->second;
	}

	Point3D pt;

	unsigned x_min=1,x_max=fieldDim.x+1;
	unsigned y_min=1,y_max=fieldDim.y+1;
	unsigned z_min=1,z_max=fieldDim.z+1;

	for (int z = 1; z < workFieldDim.z-1; z++)
		for (int y = 1; y < workFieldDim.y-1; y++)
			for (int x = 1; x < workFieldDim.x-1; x++){
				pt=Point3D(x-1,y-1,z-1);
				//             cerr<<"pt="<<pt<<" is valid "<<cellFieldG->isValid(pt)<<endl;
				///**
				currentCellPtr=cellFieldG->get(pt);
				//             currentCellPtr=cellFieldG->get(pt);
				//             cerr<<"THIS IS PTR="<<currentCellPtr<<endl;

				//             if(currentCellPtr)
				//                cerr<<"This is id="<<currentCellPtr->id<<endl;
				//currentConcentration = concentrationArray[x][y][z];

				if(secreteInMedium && ! currentCellPtr){
					concentrationField.setDirect(x,y,z,secrConstMedium);
				}

				if(currentCellPtr){
					mitr=secrData.typeIdSecrConstConstantConcentrationMap.find(currentCellPtr->type);
					if(mitr!=end_mitr){
						secrConst=mitr->second;
						concentrationField.setDirect(x,y,z,secrConst);
					}
				}
			}


}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::secrete() {

	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i ){

		for(unsigned int j = 0 ; j <diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size() ; ++j){
			(this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

			//          (this->*secrDataVec[i].secretionFcnPtrVec[j])(i);
		}


	}



}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//float ReactionDiffusionSolverFE::couplingTerm(Point3D & _pt,std::vector<CouplingData> & _couplDataVec,float _currentConcentration){
//
//	float couplingTerm=0.0;
//	float coupledConcentration;
//	for(int i =  0 ; i < _couplDataVec.size() ; ++i){
//		coupledConcentration=concentrationFieldVector[_couplDataVec[i].fieldIdx]->get(_pt);
//		couplingTerm+=_couplDataVec[i].couplingCoef*_currentConcentration*coupledConcentration;
//	}
//
//	return couplingTerm;
//
//
//
//}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::boundaryConditionInit(ConcentrationField_t *concentrationField){
	ConcentrationField_t & _array=*concentrationField;

	//dealing with periodic boundary condition in x direction
	//have to use internalDim-1 in the for loop as max index because otherwise with extra shitf if we used internalDim we run outside the lattice
	if(periodicBoundaryCheckVector[0]){//periodic boundary conditions were set in x direction	
		//x - periodic
		int x=0;
		for(int y=0 ; y< workFieldDim.y-1; ++y)
			for(int z=0 ; z<workFieldDim.z-1 ; ++z){
				_array.setDirect(x,y,z,_array.getDirect(fieldDim.x,y,z));
			}

			x=fieldDim.x+1;
			for(int y=0 ; y< workFieldDim.y-1; ++y)
				for(int z=0 ; z<workFieldDim.z-1 ; ++z){
					_array.setDirect(x,y,z,_array.getDirect(1,y,z));
				}
	}else{//noFlux BC
		int x=0;
		for(int y=0 ; y< workFieldDim.y-1; ++y)
			for(int z=0 ; z<workFieldDim.z-1 ; ++z){
				_array.setDirect(x,y,z,_array.getDirect(x+1,y,z));
			}

			x=fieldDim.x+1;
			for(int y=0 ; y< workFieldDim.y-1; ++y)
				for(int z=0 ; z<workFieldDim.z-1 ; ++z){
					_array.setDirect(x,y,z,_array.getDirect(x-1,y,z));
				}
	}

	//dealing with periodic boundary condition in y direction
	if(periodicBoundaryCheckVector[1]){//periodic boundary conditions were set in x direction
		int y=0;
		for(int x=0 ; x< workFieldDim.x-1; ++x)
			for(int z=0 ; z<workFieldDim.z-1 ; ++z){
				_array.setDirect(x,y,z,_array.getDirect(x,fieldDim.y,z));
			}

			y=fieldDim.y+1;
			for(int x=0 ; x< workFieldDim.x-1; ++x)
				for(int z=0 ; z<workFieldDim.z-1 ; ++z){
					_array.setDirect(x,y,z,_array.getDirect(x,1,z));
				}
	}else{//NoFlux BC
		int y=0;
		for(int x=0 ; x< workFieldDim.x-1; ++x)
			for(int z=0 ; z<workFieldDim.z-1 ; ++z){
				_array.setDirect(x,y,z,_array.getDirect(x,y+1,z));
			}

			y=fieldDim.y+1;
			for(int x=0 ; x< workFieldDim.x-1; ++x)
				for(int z=0 ; z<workFieldDim.z-1 ; ++z){
					_array.setDirect(x,y,z,_array.getDirect(x,y-1,z));
				}
	}

	//dealing with periodic boundary condition in z direction
	if(periodicBoundaryCheckVector[2]){//periodic boundary conditions were set in x direction
		int z=0;
		for(int x=0 ; x< workFieldDim.x-1; ++x)
			for(int y=0 ; y<workFieldDim.y-1 ; ++y){
				_array.setDirect(x,y,z,_array.getDirect(x,y,fieldDim.z));
			}

			z=fieldDim.z+1;
			for(int x=0 ; x< workFieldDim.x-1; ++x)
				for(int y=0 ; y<workFieldDim.y-1 ; ++y){
					_array.setDirect(x,y,z,_array.getDirect(x,y,1));
				}
	}else{//Noflux BC
		int z=0;
		for(int x=0 ; x< workFieldDim.x-1; ++x)
			for(int y=0 ; y<workFieldDim.y-1 ; ++y){
				_array.setDirect(x,y,z,_array.getDirect(x,y,z+1));
			}

			z=fieldDim.z+1;
			for(int x=0 ; x< workFieldDim.x-1; ++x)
				for(int y=0 ; y<workFieldDim.y-1 ; ++y){
					_array.setDirect(x,y,z,_array.getDirect(x,y,z-1));
				}
	}


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//void ReactionDiffusionSolverFE::solveRDEquations(){
//
//	/// 'n' denotes neighbor
//
//	///this is the diffusion equation
//	///C_{xx}+C_{yy}+C_{zz}=(1/a)*C_{t}
//	///a - diffusivity - diffConst
//
//	///Finite difference method:
//	///T_{0,\delta \tau}=F*\sum_{i=1}^N T_{i}+(1-N*F)*T_{0}
//	///N - number of neighbors
//	///will have to double check this formula
//
//
//	//////Point3D pt, n;
//	//////unsigned int token = 0;
//	//////double distance;
//	//////CellG * currentCellPtr=0,*nCell=0;
//
//	//////short currentCellType=0;
//	//////float concentrationSum=0.0;
//	//////float updatedConcentration=0.0;
//
//
//	//////float currentConcentration=0.0;
//	//////short neighborCounter=0;
//
//	//DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;
//	//float diffConst=diffData.diffConst;
//	vector<DiffusionData> diffDataVec(numberOfFields);
//	vector<float> diffConstVec(numberOfFields,0.0);
//	
//	bool useBoxWatcher=false;
//
//	for (int i = 0 ; i < numberOfFields ;++i){
//		diffDataVec[i]=diffSecrFieldTuppleVec[i].diffData;
//		diffConstVec[i]=diffDataVec[i].diffConst;
//		if (diffDataVec[i].useBoxWatcher){
//			useBoxWatcher=true;
//		}
//		//diffConstVec[i]=diffSecrFieldTuppleVec[i].diffData.diffConst;
//	}
//
//	//float decayConst=diffData.decayConst;
//	float dt_dx2=deltaT/(deltaX*deltaX);
//
//
//
//	std::set<unsigned char>::iterator sitr;
//	//std::set<unsigned char>::iterator end_sitr=diffData.avoidTypeIdSet.end();
//	//std::set<unsigned char>::iterator end_sitr_decay=diffData.avoidDecayInIdSet.end();
//
//	Automaton *automaton=potts->getAutomaton();
//
//	//ConcentrationField_t * concentrationFieldPtr=concentrationFieldVector[idx];
//	//ConcentrationField_t * scratchFieldPtr;
//
//	//if(!haveCouplingTerms)
//	//	scratchFieldPtr=concentrationFieldVector[diffSecrFieldTuppleVec.size()];
//	//else
//	//	scratchFieldPtr=concentrationFieldVector[diffSecrFieldTuppleVec.size()+idx];
//
//
//
//
//
//	//Array3D_t & concentrationArray = concentrationFieldPtr->getContainer();
//
//	//Array3D_t & scratchArray = scratchFieldPtr->getContainer();
//
//	for (int i = 0 ; i < numberOfFields ;++i){
//		ConcentrationField_t * concentrationFieldPtr=concentrationFieldVector[i];
//		boundaryConditionInit(concentrationFieldPtr);//initializing boundary conditions
//	}
//
//	//boundaryConditionInit(concentrationFieldPtr);//initializing boundary conditions
//
//
//
//	vector<bool> avoidMediumVec(numberOfFields,false);
//
//	for (int i = 0 ; i < numberOfFields ;++i){
//		if(diffSecrFieldTuppleVec[i].diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != diffSecrFieldTuppleVec[i].diffData.avoidTypeIdSet.end()){
//			avoidMediumVec[i]=true;
//		}
//	}
//
//	//bool avoidMedium=false;
//
//	//bool avoidDecayInMedium=false;
//	//the assumption is that medium has type ID 0
//	//if(diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != end_sitr){
//	//	avoidMedium=true;
//	//}
//
//	//if(diffData.avoidDecayInIdSet.find(automaton->getTypeId("Medium")) != end_sitr_decay){
//	//	avoidDecayInMedium=true;
//	//}
//
//	
//	if(useBoxWatcher){
//
//		unsigned x_min=1,x_max=fieldDim.x+1;
//		unsigned y_min=1,y_max=fieldDim.y+1;
//		unsigned z_min=1,z_max=fieldDim.z+1;
//
//		Dim3D minDimBW;		
//		Dim3D maxDimBW;
//		Point3D minCoordinates=*(boxWatcherSteppable->getMinCoordinatesPtr());
//		Point3D maxCoordinates=*(boxWatcherSteppable->getMaxCoordinatesPtr());
//		//cerr<<"FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates<<endl;
//		x_min=minCoordinates.x+1;
//		x_max=maxCoordinates.x+1;
//		y_min=minCoordinates.y+1;
//		y_max=maxCoordinates.y+1;
//		z_min=minCoordinates.z+1;
//		z_max=maxCoordinates.z+1;
//
//		minDimBW=Dim3D(x_min,y_min,z_min);
//		maxDimBW=Dim3D(x_max,y_max,z_max);
//		pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW,maxDimBW);
//
//	}
//
//
////managing number of threads has to be done BEFORE parallel section otherwise undefined behavior will occur
//pUtils->prepareParallelRegionFESolvers(useBoxWatcher);
//#pragma omp parallel
//	{	
//
//
//
//	CellG *currentCellPtr=0;
//	Point3D pt, n;
//	float currentConcentration=0;
//	float updatedConcentration=0.0;
//	float concentrationSum=0.0;
//	short neighborCounter=0;
//	CellG *neighborCellPtr=0;
//
//		try
//		{
//
//
//
//		int threadNumber=pUtils->getCurrentWorkNodeNumber();
//
//		Dim3D minDim;		
//		Dim3D maxDim;
//
//		if(useBoxWatcher){
//			minDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
//			maxDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;
//
//		}else{
//			minDim=pUtils->getFESolverPartition(threadNumber).first;
//			maxDim=pUtils->getFESolverPartition(threadNumber).second;
//		}
//
//
//
//		for (int z = minDim.z; z < maxDim.z; z++)
//			for (int y = minDim.y; y < maxDim.y; y++)
//				for (int x = minDim.x; x < maxDim.x; x++){
//
//
//
//						pt=Point3D(x-1,y-1,z-1);
//						///**
//						currentCellPtr=cellFieldG->getQuick(pt);
//						if (currentCellPtr)
//							variableCellTypeMu[threadNumber]=currentCellPtr->type;
//						else
//							variableCellTypeMu[threadNumber]=0;
//
//						//          currentCellPtr=cellFieldG->get(pt);
//
//						//getting concentrations at x,y,z for all the fields	
//						for (int fieldIdx=0 ; fieldIdx<numberOfFields; ++fieldIdx){						
//							ConcentrationField_t & concentrationField = *concentrationFieldVector[fieldIdx];
//							variableConcentrationVecMu[threadNumber][fieldIdx]=concentrationField.getDirect(x,y,z);
//							//if (x==1&&y==1){
//							//	cerr<<"Field "<<fieldIdx<<" c="<<variableConcentrationVecMu[fieldIdx]<<endl;
//							//}
//
//						}
//						// finding concentrations at x,y,z at t+dt
//						for (int fieldIdx=0 ; fieldIdx<numberOfFields; ++fieldIdx){
//							//for (int z = 1; z < workFieldDim.z-1; z++)
//							// for (int y = 1; y < workFieldDim.y-1; y++)
//							//   for (int x = 1; x < workFieldDim.x-1; x++){
//
//
//
//							ConcentrationField_t & concentrationField = *concentrationFieldVector[fieldIdx];
//							currentConcentration = concentrationField.getDirect(x,y,z);
//							//DiffusionData & diffData = diffSecrFieldTuppleVec[fieldIdx].diffData;
//							//float diffConst=diffData.diffConst;
//
//							//DoNotDiffuseTo means do not solve RD equations for points occupied by this cell type
//							if(avoidMediumVec[fieldIdx] && !currentCellPtr){//if medium is to be avoided
//								//scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								concentrationField.setDirectSwap(x,y,z,variableConcentrationVecMu[threadNumber][fieldIdx]);
//								//if(avoidDecayInMedium){
//								//	scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								//	//scratchArray[x][y][z]=currentConcentration;
//								//}
//								//else{
//								//	scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								//	//scratchArray[x][y][z]=currentConcentration-deltaT*(decayConst*currentConcentration);
//								//}
//								continue;
//							}
//
//							if(currentCellPtr && diffDataVec[fieldIdx].avoidTypeIdSet.find(currentCellPtr->type)!=diffDataVec[fieldIdx].avoidTypeIdSet.end()){
//								//scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								concentrationField.setDirectSwap(x,y,z,variableConcentrationVecMu[threadNumber][fieldIdx]);
//								//if(diffData.avoidDecayInIdSet.find(currentCellPtr->type)!=end_sitr_decay){
//								//	scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								//	//scratchArray[x][y][z]=currentConcentration;
//								//}else{
//								//	scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								//	//scratchArray[x][y][z]=currentConcentration-deltaT*(decayConst*currentConcentration);
//								//}
//								continue; // avoid user defined types
//							}
//
//							updatedConcentration=0.0;
//							concentrationSum=0.0;
//							neighborCounter=0;
//
//							//loop over nearest neighbors
//							CellG *neighborCellPtr=0;
//							const std::vector<Point3D> & offsetVecRef=boundaryStrategy->getOffsetVec(pt);
//							//          cerr<<"maxNeighborIndex="<<maxNeighborIndex<<endl;
//
//							for (int i = 0  ; i<=maxNeighborIndex /*offsetVec.size()*/ ; ++i ){ 
//								const Point3D & offset = offsetVecRef[i];
//
//
//								if(diffDataVec[fieldIdx].avoidTypeIdSet.size()||avoidMediumVec[fieldIdx]){ //means user defined types to avoid in terms of the diffusion
//
//									n=pt+offsetVecRef[i];
//									neighborCellPtr=cellFieldG->get(n);
//									if(avoidMediumVec[fieldIdx] && !neighborCellPtr) continue; // avoid medium if specified by the user
//									if(neighborCellPtr && diffDataVec[fieldIdx].avoidTypeIdSet.find(neighborCellPtr->type)!=diffDataVec[fieldIdx].avoidTypeIdSet.end()) continue;//avoid user specified types
//								}
//								concentrationSum += concentrationField.getDirect(x+offset.x,y+offset.y,z+offset.z);
//
//								++neighborCounter;
//
//							}
//
//
//							updatedConcentration =  dt_dx2*diffConstVec[fieldIdx]*(concentrationSum - neighborCounter*currentConcentration)+currentConcentration;
//
//
//							//processing decay depandent on type of the current cell
//							//if(currentCellPtr){
//							//	if(diffData.avoidDecayInIdSet.find(currentCellPtr->type)!=end_sitr_decay){
//							//		;//decay in this type is forbidden
//							//	}else{
//							//		updatedConcentration-=deltaT*(decayConst*currentConcentration);//decay in this type is allowed
//							//	}
//							//}else{
//							//	if(avoidDecayInMedium){
//							//		;//decay in Medium is forbidden
//							//	}else{
//							//		updatedConcentration-=deltaT*(decayConst*currentConcentration); //decay in Medium is allowed
//							//	}
//							//}
//							//if(haveCouplingTerms){
//							//	updatedConcentration+=couplingTerm(pt,diffData.couplingDataVec,currentConcentration);
//							//}
//
//							//additionalTerm contributions
//							
//							
//							
//							
//							updatedConcentration+=deltaT*parserVec[threadNumber][fieldIdx].Eval();
//
//
//
//
//							//if(parserVec[fieldIdx].GetExpr() != ""){
//							//	updatedConcentration+=deltaT*parserVec[fieldIdx].Eval();
//							//	//if (x==1&&y==1){
//							//	//	cerr<<"Field "<<fieldIdx<<" parserVec[fieldIdx].Eval()="<<parserVec[fieldIdx].Eval()<<endl;
//							//	//}
//							//}
//
//
//							//imposing artificial limits on allowed concentration
//							//if(diffData.useThresholds){
//							//	if(updatedConcentration>diffData.maxConcentration){
//							//		updatedConcentration=diffData.maxConcentration;
//							//	}
//							//	if(updatedConcentration<diffData.minConcentration){
//							//		updatedConcentration=diffData.minConcentration;
//							//	}
//							//}
//
//
//							concentrationField.setDirectSwap(x,y,z,updatedConcentration);//updating scratch
//
//
//						}
//					}
//
//					//for (int fieldIdx=0 ; fieldIdx<numberOfFields; ++fieldIdx){
//					//	ConcentrationField_t & concentrationField = *concentrationFieldVector[fieldIdx];
//					//	concentrationField.swapArrays();
//					//}
//
//		} catch (mu::Parser::exception_type &e)
//		{
//			cerr<<e.GetMsg()<<endl;
//			ASSERT_OR_THROW(e.GetMsg(),0);
//		}
//	}
//
//	for (int fieldIdx=0 ; fieldIdx<numberOfFields; ++fieldIdx){
//		ConcentrationField_t & concentrationField = *concentrationFieldVector[fieldIdx];
//		concentrationField.swapArrays();
//	}
//	
//}


//void ReactionDiffusionSolverFE::solveRDEquations(){
//
//	/// 'n' denotes neighbor
//
//	///this is the diffusion equation
//	///C_{xx}+C_{yy}+C_{zz}=(1/a)*C_{t}
//	///a - diffusivity - diffConst
//
//	///Finite difference method:
//	///T_{0,\delta \tau}=F*\sum_{i=1}^N T_{i}+(1-N*F)*T_{0}
//	///N - number of neighbors
//	///will have to double check this formula
//
//
//	//////Point3D pt, n;
//	//////unsigned int token = 0;
//	//////double distance;
//	//////CellG * currentCellPtr=0,*nCell=0;
//
//	//////short currentCellType=0;
//	//////float concentrationSum=0.0;
//	//////float updatedConcentration=0.0;
//
//
//	//////float currentConcentration=0.0;
//	//////short neighborCounter=0;
//
//	//DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;
//	//float diffConst=diffData.diffConst;
//	vector<DiffusionData> diffDataVec(numberOfFields);
//	vector<float> diffConstVec(numberOfFields,0.0);
//	
//	bool useBoxWatcher=false;
//
//	for (int i = 0 ; i < numberOfFields ;++i){
//		diffDataVec[i]=diffSecrFieldTuppleVec[i].diffData;
//		diffConstVec[i]=diffDataVec[i].diffConst;
//		if (diffDataVec[i].useBoxWatcher){
//			useBoxWatcher=true;
//		}
//		//diffConstVec[i]=diffSecrFieldTuppleVec[i].diffData.diffConst;
//	}
//
//	//float decayConst=diffData.decayConst;
//	float dt_dx2=deltaT/(deltaX*deltaX);
//
//
//
//	std::set<unsigned char>::iterator sitr;
//	//std::set<unsigned char>::iterator end_sitr=diffData.avoidTypeIdSet.end();
//	//std::set<unsigned char>::iterator end_sitr_decay=diffData.avoidDecayInIdSet.end();
//
//	Automaton *automaton=potts->getAutomaton();
//
//	//ConcentrationField_t * concentrationFieldPtr=concentrationFieldVector[idx];
//	//ConcentrationField_t * scratchFieldPtr;
//
//	//if(!haveCouplingTerms)
//	//	scratchFieldPtr=concentrationFieldVector[diffSecrFieldTuppleVec.size()];
//	//else
//	//	scratchFieldPtr=concentrationFieldVector[diffSecrFieldTuppleVec.size()+idx];
//
//
//
//
//
//	//Array3D_t & concentrationArray = concentrationFieldPtr->getContainer();
//
//	//Array3D_t & scratchArray = scratchFieldPtr->getContainer();
//
//	for (int i = 0 ; i < numberOfFields ;++i){
//		ConcentrationField_t * concentrationFieldPtr=concentrationFieldVector[i];
//		boundaryConditionInit(concentrationFieldPtr);//initializing boundary conditions
//	}
//
//	//boundaryConditionInit(concentrationFieldPtr);//initializing boundary conditions
//
//
//
//	vector<bool> avoidMediumVec(numberOfFields,false);
//
//	for (int i = 0 ; i < numberOfFields ;++i){
//		if(diffSecrFieldTuppleVec[i].diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != diffSecrFieldTuppleVec[i].diffData.avoidTypeIdSet.end()){
//			avoidMediumVec[i]=true;
//		}
//	}
//
//	//bool avoidMedium=false;
//
//	//bool avoidDecayInMedium=false;
//	//the assumption is that medium has type ID 0
//	//if(diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != end_sitr){
//	//	avoidMedium=true;
//	//}
//
//	//if(diffData.avoidDecayInIdSet.find(automaton->getTypeId("Medium")) != end_sitr_decay){
//	//	avoidDecayInMedium=true;
//	//}
//
//	
//	if(useBoxWatcher){
//
//		unsigned x_min=1,x_max=fieldDim.x+1;
//		unsigned y_min=1,y_max=fieldDim.y+1;
//		unsigned z_min=1,z_max=fieldDim.z+1;
//
//		Dim3D minDimBW;		
//		Dim3D maxDimBW;
//		Point3D minCoordinates=*(boxWatcherSteppable->getMinCoordinatesPtr());
//		Point3D maxCoordinates=*(boxWatcherSteppable->getMaxCoordinatesPtr());
//		//cerr<<"FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates<<endl;
//		x_min=minCoordinates.x+1;
//		x_max=maxCoordinates.x+1;
//		y_min=minCoordinates.y+1;
//		y_max=maxCoordinates.y+1;
//		z_min=minCoordinates.z+1;
//		z_max=maxCoordinates.z+1;
//
//		minDimBW=Dim3D(x_min,y_min,z_min);
//		maxDimBW=Dim3D(x_max,y_max,z_max);
//		pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW,maxDimBW);
//
//	}
//
//
////managing number of threads has to be done BEFORE parallel section otherwise undefined behavior will occur
//pUtils->prepareParallelRegionFESolvers(useBoxWatcher);
//#pragma omp parallel
//	{	
//
//
//
//	CellG *currentCellPtr=0;
//	Point3D pt, n;
//	float currentConcentration=0;
//	float updatedConcentration=0.0;
//	float concentrationSum=0.0;
//	short neighborCounter=0;
//	CellG *neighborCellPtr=0;
//
//		try
//		{
//
//
//
//		int threadNumber=pUtils->getCurrentWorkNodeNumber();
//
//		Dim3D minDim;		
//		Dim3D maxDim;
//
//		if(useBoxWatcher){
//			minDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
//			maxDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;
//
//		}else{
//			minDim=pUtils->getFESolverPartition(threadNumber).first;
//			maxDim=pUtils->getFESolverPartition(threadNumber).second;
//		}
//
//
//		// finding concentrations at x,y,z at t+dt
//		for (int fIdx=0 ; fIdx<numberOfFields; ++fIdx){
//			ConcentrationField_t & concentrationField = *concentrationFieldVector[fIdx];
//			
//		for (int z = minDim.z; z < maxDim.z; z++)
//			for (int y = minDim.y; y < maxDim.y; y++)
//				for (int x = minDim.x; x < maxDim.x; x++){
//
//						pt=Point3D(x-1,y-1,z-1);
//						///**
//						currentCellPtr=cellFieldG->getQuick(pt);
//						currentConcentration = concentrationField.getDirect(x,y,z);
//
//						if (currentCellPtr)
//							variableCellTypeMu[threadNumber]=currentCellPtr->type;
//						else
//							variableCellTypeMu[threadNumber]=0;
//
//						//getting concentrations at x,y,z for all the fields	
//						for (int fieldIdx=0 ; fieldIdx<numberOfFields; ++fieldIdx){						
//							ConcentrationField_t & concentrationField = *concentrationFieldVector[fieldIdx];
//							variableConcentrationVecMu[threadNumber][fieldIdx]=concentrationField.getDirect(x,y,z);
//
//						}
//
//
//							//DoNotDiffuseTo means do not solve RD equations for points occupied by this cell type
//							if(avoidMediumVec[fIdx] && !currentCellPtr){//if medium is to be avoided
//								//scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								concentrationField.setDirectSwap(x,y,z,variableConcentrationVecMu[threadNumber][fIdx]);
//								//if(avoidDecayInMedium){
//								//	scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								//	//scratchArray[x][y][z]=currentConcentration;
//								//}
//								//else{
//								//	scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								//	//scratchArray[x][y][z]=currentConcentration-deltaT*(decayConst*currentConcentration);
//								//}
//								continue;
//							}
//
//							if(currentCellPtr && diffDataVec[fIdx].avoidTypeIdSet.find(currentCellPtr->type)!=diffDataVec[fIdx].avoidTypeIdSet.end()){
//								//scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								concentrationField.setDirectSwap(x,y,z,variableConcentrationVecMu[threadNumber][fIdx]);
//								//if(diffData.avoidDecayInIdSet.find(currentCellPtr->type)!=end_sitr_decay){
//								//	scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								//	//scratchArray[x][y][z]=currentConcentration;
//								//}else{
//								//	scratchArray[x][y][z]=variableConcentrationVecMu[fieldIdx];
//								//	//scratchArray[x][y][z]=currentConcentration-deltaT*(decayConst*currentConcentration);
//								//}
//								continue; // avoid user defined types
//							}
//
//							updatedConcentration=0.0;
//							concentrationSum=0.0;
//							neighborCounter=0;
//
//							//loop over nearest neighbors
//							CellG *neighborCellPtr=0;
//							const std::vector<Point3D> & offsetVecRef=boundaryStrategy->getOffsetVec(pt);
//							//          cerr<<"maxNeighborIndex="<<maxNeighborIndex<<endl;
//
//							for (int i = 0  ; i<=maxNeighborIndex /*offsetVec.size()*/ ; ++i ){ 
//								const Point3D & offset = offsetVecRef[i];
//
//
//								if(diffDataVec[fIdx].avoidTypeIdSet.size()||avoidMediumVec[fIdx]){ //means user defined types to avoid in terms of the diffusion
//
//									n=pt+offsetVecRef[i];
//									neighborCellPtr=cellFieldG->get(n);
//									if(avoidMediumVec[fIdx] && !neighborCellPtr) continue; // avoid medium if specified by the user
//									if(neighborCellPtr && diffDataVec[fIdx].avoidTypeIdSet.find(neighborCellPtr->type)!=diffDataVec[fIdx].avoidTypeIdSet.end()) continue;//avoid user specified types
//								}
//								concentrationSum += concentrationField.getDirect(x+offset.x,y+offset.y,z+offset.z);
//
//								++neighborCounter;
//
//							}
//
//
//							updatedConcentration =  dt_dx2*diffConstVec[fIdx]*(concentrationSum - neighborCounter*currentConcentration)+currentConcentration;
//
//
//							//processing decay depandent on type of the current cell
//							//if(currentCellPtr){
//							//	if(diffData.avoidDecayInIdSet.find(currentCellPtr->type)!=end_sitr_decay){
//							//		;//decay in this type is forbidden
//							//	}else{
//							//		updatedConcentration-=deltaT*(decayConst*currentConcentration);//decay in this type is allowed
//							//	}
//							//}else{
//							//	if(avoidDecayInMedium){
//							//		;//decay in Medium is forbidden
//							//	}else{
//							//		updatedConcentration-=deltaT*(decayConst*currentConcentration); //decay in Medium is allowed
//							//	}
//							//}
//							//if(haveCouplingTerms){
//							//	updatedConcentration+=couplingTerm(pt,diffData.couplingDataVec,currentConcentration);
//							//}
//
//							//additionalTerm contributions
//							
//							
//							
//							
//							updatedConcentration+=deltaT*parserVec[threadNumber][fIdx].Eval();
//
//
//
//
//							//if(parserVec[fieldIdx].GetExpr() != ""){
//							//	updatedConcentration+=deltaT*parserVec[fieldIdx].Eval();
//							//	//if (x==1&&y==1){
//							//	//	cerr<<"Field "<<fieldIdx<<" parserVec[fieldIdx].Eval()="<<parserVec[fieldIdx].Eval()<<endl;
//							//	//}
//							//}
//
//
//							//imposing artificial limits on allowed concentration
//							//if(diffData.useThresholds){
//							//	if(updatedConcentration>diffData.maxConcentration){
//							//		updatedConcentration=diffData.maxConcentration;
//							//	}
//							//	if(updatedConcentration<diffData.minConcentration){
//							//		updatedConcentration=diffData.minConcentration;
//							//	}
//							//}
//
//
//							concentrationField.setDirectSwap(x,y,z,updatedConcentration);//updating scratch
//
//
//						}
//					}
//
//					//for (int fieldIdx=0 ; fieldIdx<numberOfFields; ++fieldIdx){
//					//	ConcentrationField_t & concentrationField = *concentrationFieldVector[fieldIdx];
//					//	concentrationField.swapArrays();
//					//}
//
//		} catch (mu::Parser::exception_type &e)
//		{
//			cerr<<e.GetMsg()<<endl;
//			ASSERT_OR_THROW(e.GetMsg(),0);
//		}
//	}
//
//	for (int fieldIdx=0 ; fieldIdx<numberOfFields; ++fieldIdx){
//		ConcentrationField_t & concentrationField = *concentrationFieldVector[fieldIdx];
//		concentrationField.swapArrays();
//	}
//	
//}


void ReactionDiffusionSolverFE::solveRDEquations(){



	for (int idx=0 ; idx<numberOfFields; ++idx){
		solveRDEquationsSingleField(idx);
	}

	for (int fieldIdx=0 ; fieldIdx<numberOfFields; ++fieldIdx){
		ConcentrationField_t & concentrationField = *concentrationFieldVector[fieldIdx];
		concentrationField.swapArrays();
	}
	
}

void ReactionDiffusionSolverFE::solveRDEquationsSingleField(unsigned int idx){

	/// 'n' denotes neighbor

	///this is the diffusion equation
	///C_{xx}+C_{yy}+C_{zz}=(1/a)*C_{t}
	///a - diffusivity - diffConst

	///Finite difference method:
	///T_{0,\delta \tau}=F*\sum_{i=1}^N T_{i}+(1-N*F)*T_{0}
	///N - number of neighbors
	///will have to double check this formula



	DiffusionData diffData=diffSecrFieldTuppleVec[idx].diffData;
	float diffConst=diffData.diffConst;
	
	bool useBoxWatcher=false;

	if (diffData.useBoxWatcher)
		useBoxWatcher=true;




	float dt_dx2=deltaT/(deltaX*deltaX);



	std::set<unsigned char>::iterator sitr;


	Automaton *automaton=potts->getAutomaton();


	ConcentrationField_t * concentrationFieldPtr=concentrationFieldVector[idx];
	boundaryConditionInit(concentrationFieldPtr);//initializing boundary conditions



	//////vector<bool> avoidMediumVec(numberOfFields,false);
	bool avoidMedium=false;
	if(diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != diffData.avoidTypeIdSet.end()){
		avoidMedium=true;
	}

	
	if(useBoxWatcher){

		unsigned x_min=1,x_max=fieldDim.x+1;
		unsigned y_min=1,y_max=fieldDim.y+1;
		unsigned z_min=1,z_max=fieldDim.z+1;

		Dim3D minDimBW;		
		Dim3D maxDimBW;
		Point3D minCoordinates=*(boxWatcherSteppable->getMinCoordinatesPtr());
		Point3D maxCoordinates=*(boxWatcherSteppable->getMaxCoordinatesPtr());
		//cerr<<"FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates<<endl;
		x_min=minCoordinates.x+1;
		x_max=maxCoordinates.x+1;
		y_min=minCoordinates.y+1;
		y_max=maxCoordinates.y+1;
		z_min=minCoordinates.z+1;
		z_max=maxCoordinates.z+1;

		minDimBW=Dim3D(x_min,y_min,z_min);
		maxDimBW=Dim3D(x_max,y_max,z_max);
		pUtils->calculateFESolverPartitionWithBoxWatcher(minDimBW,maxDimBW);

	}


//managing number of threads has to be done BEFORE parallel section otherwise undefined behavior will occur
pUtils->prepareParallelRegionFESolvers(useBoxWatcher);
#pragma omp parallel
	{	



	CellG *currentCellPtr=0;
	Point3D pt, n;
	float currentConcentration=0;
	float updatedConcentration=0.0;
	float concentrationSum=0.0;
	short neighborCounter=0;
	CellG *neighborCellPtr=0;

		try
		{



		int threadNumber=pUtils->getCurrentWorkNodeNumber();

		Dim3D minDim;		
		Dim3D maxDim;

		if(useBoxWatcher){
			minDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
			maxDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

		}else{
			minDim=pUtils->getFESolverPartition(threadNumber).first;
			maxDim=pUtils->getFESolverPartition(threadNumber).second;
		}


		// finding concentrations at x,y,z at t+dt
		
		ConcentrationField_t & concentrationField = *concentrationFieldVector[idx];
			
		for (int z = minDim.z; z < maxDim.z; z++)
			for (int y = minDim.y; y < maxDim.y; y++)
				for (int x = minDim.x; x < maxDim.x; x++){

						pt=Point3D(x-1,y-1,z-1);
						///**
						currentCellPtr=cellFieldG->getQuick(pt);
						currentConcentration = concentrationField.getDirect(x,y,z);

						if (currentCellPtr)
							variableCellTypeMu[threadNumber]=currentCellPtr->type;
						else
							variableCellTypeMu[threadNumber]=0;

						//getting concentrations at x,y,z for all the fields	
						for (int fieldIdx=0 ; fieldIdx<numberOfFields; ++fieldIdx){						
							ConcentrationField_t & concentrationField = *concentrationFieldVector[fieldIdx];
							variableConcentrationVecMu[threadNumber][fieldIdx]=concentrationField.getDirect(x,y,z);

						}


							//DoNotDiffuseTo means do not solve RD equations for points occupied by this cell type
							if(avoidMedium && !currentCellPtr){//if medium is to be avoided
								concentrationField.setDirectSwap(x,y,z,variableConcentrationVecMu[threadNumber][idx]);
								continue;
							}

							if(currentCellPtr && diffData.avoidTypeIdSet.find(currentCellPtr->type)!=diffData.avoidTypeIdSet.end()){

								concentrationField.setDirectSwap(x,y,z,variableConcentrationVecMu[threadNumber][idx]);
								continue; // avoid user defined types
							}

							updatedConcentration=0.0;
							concentrationSum=0.0;
							neighborCounter=0;

							//loop over nearest neighbors
							CellG *neighborCellPtr=0;
							const std::vector<Point3D> & offsetVecRef=boundaryStrategy->getOffsetVec(pt);
							//          cerr<<"maxNeighborIndex="<<maxNeighborIndex<<endl;

							for (int i = 0  ; i<=maxNeighborIndex /*offsetVec.size()*/ ; ++i ){ 
								const Point3D & offset = offsetVecRef[i];


								if(diffData.avoidTypeIdSet.size()||avoidMedium){ //means user defined types to avoid in terms of the diffusion

									n=pt+offsetVecRef[i];
									neighborCellPtr=cellFieldG->get(n);
									if(avoidMedium && !neighborCellPtr) continue; // avoid medium if specified by the user
									if(neighborCellPtr && diffData.avoidTypeIdSet.find(neighborCellPtr->type)!=diffData.avoidTypeIdSet.end()) continue;//avoid user specified types
								}
								concentrationSum += concentrationField.getDirect(x+offset.x,y+offset.y,z+offset.z);

								++neighborCounter;

							}


							updatedConcentration =  dt_dx2*diffConst*(concentrationSum - neighborCounter*currentConcentration)+currentConcentration;

							//additionalTerm contributions
							
							
							updatedConcentration+=deltaT*parserVec[threadNumber][idx].Eval();






							concentrationField.setDirectSwap(x,y,z,updatedConcentration);//updating scratch


						}
					


		} catch (mu::Parser::exception_type &e)
		{
			cerr<<e.GetMsg()<<endl;
			ASSERT_OR_THROW(e.GetMsg(),0);
		}
	}


}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool ReactionDiffusionSolverFE::isBoudaryRegion(int x, int y, int z, Dim3D dim)
{
	if (x < 2 || x > dim.x - 3 || y < 2 || y > dim.y - 3 || z < 2 || z > dim.z - 3 )
		return true;
	else
		return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::diffuse() {



	solveRDEquations();


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::scrarch2Concentration( ConcentrationField_t *scratchField, ConcentrationField_t *concentrationField){
	//scratchField->switchContainersQuick(*(concentrationField));

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::outputField( std::ostream & _out, ConcentrationField_t *_concentrationField){
	Point3D pt;
	float tempValue;

	for (pt.z = 0; pt.z < fieldDim.z; pt.z++)
		for (pt.y = 0; pt.y < fieldDim.y; pt.y++)
			for (pt.x = 0; pt.x < fieldDim.x; pt.x++){
				tempValue=_concentrationField->get(pt);
				_out<<pt.x<<" "<<pt.y<<" "<<pt.z<<" "<<tempValue<<endl;
			}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReactionDiffusionSolverFE::readConcentrationField(std::string fileName,ConcentrationField_t *concentrationField){

	ifstream in(fileName.c_str());

	ASSERT_OR_THROW(string("Could not open chemical concentration file '") +
		fileName + "'!", in.is_open());

	Point3D pt;
	float c;
	//Zero entire field
	for (pt.z = 0; pt.z < fieldDim.z; pt.z++)
		for (pt.y = 0; pt.y < fieldDim.y; pt.y++)
			for (pt.x = 0; pt.x < fieldDim.x; pt.x++){
				concentrationField->set(pt,0);
			}

			while(!in.eof()){
				in>>pt.x>>pt.y>>pt.z>>c;
				if(!in.fail())
					concentrationField->set(pt,c);
			}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ReactionDiffusionSolverFE::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	if(potts->getDisplayUnitsFlag()){
		Unit diffConstUnit=powerUnit(potts->getLengthUnit(),2)/potts->getTimeUnit();
		Unit decayConstUnit=1/potts->getTimeUnit();
		Unit secretionConstUnit=1/potts->getTimeUnit();

		CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
		if (!unitsElem){ //add Units element
			unitsElem=_xmlData->attachElement("Units");
		}

		if(unitsElem->getFirstElement("DiffusionConstantUnit")){
			unitsElem->getFirstElement("DiffusionConstantUnit")->updateElementValue(diffConstUnit.toString());
		}else{
			unitsElem->attachElement("DiffusionConstantUnit",diffConstUnit.toString());
		}

		if(unitsElem->getFirstElement("DecayConstantUnit")){
			unitsElem->getFirstElement("DecayConstantUnit")->updateElementValue(decayConstUnit.toString());
		}else{
			unitsElem->attachElement("DecayConstantUnit",decayConstUnit.toString());
		}

		if(unitsElem->getFirstElement("DeltaXUnit")){
			unitsElem->getFirstElement("DeltaXUnit")->updateElementValue(potts->getLengthUnit().toString());
		}else{
			unitsElem->attachElement("DeltaXUnit",potts->getLengthUnit().toString());
		}

		if(unitsElem->getFirstElement("DeltaTUnit")){
			unitsElem->getFirstElement("DeltaTUnit")->updateElementValue(potts->getTimeUnit().toString());
		}else{
			unitsElem->attachElement("DeltaTUnit",potts->getTimeUnit().toString());
		}

		if(unitsElem->getFirstElement("CouplingCoefficientUnit")){
			unitsElem->getFirstElement("CouplingCoefficientUnit")->updateElementValue(decayConstUnit.toString());
		}else{
			unitsElem->attachElement("CouplingCoefficientUnit",decayConstUnit.toString());
		}



		if(unitsElem->getFirstElement("SecretionUnit")){
			unitsElem->getFirstElement("SecretionUnit")->updateElementValue(secretionConstUnit.toString());
		}else{
			unitsElem->attachElement("SecretionUnit",secretionConstUnit.toString());
		}

		if(unitsElem->getFirstElement("SecretionOnContactUnit")){
			unitsElem->getFirstElement("SecretionOnContactUnit")->updateElementValue(secretionConstUnit.toString());
		}else{
			unitsElem->attachElement("SecretionOnContactUnit",secretionConstUnit.toString());
		}

		if(unitsElem->getFirstElement("ConstantConcentrationUnit")){
			unitsElem->getFirstElement("ConstantConcentrationUnit")->updateElementValue(secretionConstUnit.toString());
		}else{
			unitsElem->attachElement("ConstantConcentrationUnit",secretionConstUnit.toString());
		}

		if(unitsElem->getFirstElement("DecayConstantUnit")){
			unitsElem->getFirstElement("DecayConstantUnit")->updateElementValue(decayConstUnit.toString());
		}else{
			unitsElem->attachElement("DecayConstantUnit",decayConstUnit.toString());
		}

		if(unitsElem->getFirstElement("DeltaXUnit")){
			unitsElem->getFirstElement("DeltaXUnit")->updateElementValue(potts->getLengthUnit().toString());
		}else{
			unitsElem->attachElement("DeltaXUnit",potts->getLengthUnit().toString());
		}

		if(unitsElem->getFirstElement("DeltaTUnit")){
			unitsElem->getFirstElement("DeltaTUnit")->updateElementValue(potts->getTimeUnit().toString());
		}else{
			unitsElem->attachElement("DeltaTUnit",potts->getTimeUnit().toString());
		}

		if(unitsElem->getFirstElement("CouplingCoefficientUnit")){
			unitsElem->getFirstElement("CouplingCoefficientUnit")->updateElementValue(decayConstUnit.toString());
		}else{
			unitsElem->attachElement("CouplingCoefficientUnit",decayConstUnit.toString());
		}

		if(unitsElem->getFirstElement("UptakeUnit")){
			unitsElem->getFirstElement("UptakeUnit")->updateElementValue(decayConstUnit.toString());
		}else{
			unitsElem->attachElement("UptakeUnit",decayConstUnit.toString());
		}

		if(unitsElem->getFirstElement("RelativeUptakeUnit")){
			unitsElem->getFirstElement("RelativeUptakeUnit")->updateElementValue(decayConstUnit.toString());
		}else{
			unitsElem->attachElement("RelativeUptakeUnit",decayConstUnit.toString());
		}

		if(unitsElem->getFirstElement("MaxUptakeUnit")){
			unitsElem->getFirstElement("MaxUptakeUnit")->updateElementValue(decayConstUnit.toString());
		}else{
			unitsElem->attachElement("MaxUptakeUnit",decayConstUnit.toString());
		}



	}

	//notice, only basic steering is enabled for PDE solvers - changing diffusion constants, do -not-diffuse to types etc...
	// Coupling coefficients cannot be changed and also there is no way to allocate extra fields while simulation is running

	diffSecrFieldTuppleVec.clear();

	if(_xmlData->findElement("DeltaX"))
		deltaX=_xmlData->getFirstElement("DeltaX")->getDouble();

	if(_xmlData->findElement("DeltaT"))
		deltaT=_xmlData->getFirstElement("DeltaT")->getDouble();

	if(_xmlData->findElement("CellTypeVariableName"))
		cellTypeVariableName=_xmlData->getFirstElement("CellTypeVariableName")->getDouble();
	if(_xmlData->findElement("UseBoxWatcher"))
		useBoxWatcher=true;

	CC3DXMLElementList diffFieldXMLVec=_xmlData->getElements("DiffusionField");
	for(unsigned int i = 0 ; i < diffFieldXMLVec.size() ; ++i ){
		diffSecrFieldTuppleVec.push_back(DiffusionSecretionRDFieldTupple());
		DiffusionData & diffData=diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size()-1].diffData;
		SecretionData & secrData=diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size()-1].secrData;

		if(diffFieldXMLVec[i]->findElement("DiffusionData"))
			diffData.update(diffFieldXMLVec[i]->getFirstElement("DiffusionData"));

		if(diffFieldXMLVec[i]->findElement("SecretionData"))
			secrData.update(diffFieldXMLVec[i]->getFirstElement("SecretionData"));

		if(diffFieldXMLVec[i]->findElement("ReadFromFile"))
			readFromFileFlag=true;


	}
	if(_xmlData->findElement("Serialize")){

		serializeFlag=true;
		if(_xmlData->getFirstElement("Serialize")->findAttribute("Frequency")){
			serializeFrequency=_xmlData->getFirstElement("Serialize")->getAttributeAsUInt("Frequency");
		}
		cerr<<"serialize Flag="<<serializeFlag<<endl;

	}

	if(_xmlData->findElement("ReadFromFile")){
		readFromFileFlag=true;
		cerr<<"readFromFileFlag="<<readFromFileFlag<<endl;
	}


	for(int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		diffSecrFieldTuppleVec[i].diffData.setAutomaton(automaton);
		diffSecrFieldTuppleVec[i].secrData.setAutomaton(automaton);
		diffSecrFieldTuppleVec[i].diffData.initialize(automaton);
		diffSecrFieldTuppleVec[i].secrData.initialize(automaton);
	}

	///assigning member method ptrs to the vector

	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){

		diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.assign(diffSecrFieldTuppleVec[i].secrData.secrTypesNameSet.size(),0);
		unsigned int j=0;
		for(set<string>::iterator sitr=diffSecrFieldTuppleVec[i].secrData.secrTypesNameSet.begin() ; sitr != diffSecrFieldTuppleVec[i].secrData.secrTypesNameSet.end()  ; ++sitr){

			if((*sitr)=="Secretion"){
				diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&ReactionDiffusionSolverFE::secreteSingleField;
				++j;
			}
			else if((*sitr)=="SecretionOnContact"){
				diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&ReactionDiffusionSolverFE::secreteOnContactSingleField;
				++j;
			}
			else if((*sitr)=="ConstantConcentration"){
				diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&ReactionDiffusionSolverFE::secreteConstantConcentrationSingleField;
				++j;
			}

		}
	}

	numberOfFields=diffSecrFieldTuppleVec.size();
	//allocate vector of parsers
	parserVec.assign(pUtils->getMaxNumberOfWorkNodesFESolver(),vector<mu::Parser>(numberOfFields,mu::Parser()));

	variableConcentrationVecMu.assign(pUtils->getMaxNumberOfWorkNodesFESolver(),vector<double>(numberOfFields,0.0));
	variableCellTypeMu.assign(pUtils->getMaxNumberOfWorkNodesFESolver(),0.0);
	//initialize parsers
	try{
		for(unsigned int t = 0 ; t < pUtils->getMaxNumberOfWorkNodesFESolver(); ++t){
		for(unsigned int i = 0 ; i < numberOfFields ; ++i){
			for(unsigned int j = 0 ; j < numberOfFields ; ++j){
				parserVec[t][i].DefineVar(diffSecrFieldTuppleVec[j].diffData.fieldName, &variableConcentrationVecMu[t][j]);
			}
			parserVec[t][i].DefineVar(cellTypeVariableName,&variableCellTypeMu[t]);
			if (diffSecrFieldTuppleVec[i].diffData.additionalTerm==""){
				diffSecrFieldTuppleVec[i].diffData.additionalTerm="0.0"; //in case additonal term is set empty we will return 0.0
			}
			parserVec[t][i].SetExpr(diffSecrFieldTuppleVec[i].diffData.additionalTerm);
		}
		}

	} catch (mu::Parser::exception_type &e)
	{
		cerr<<e.GetMsg()<<endl;
		ASSERT_OR_THROW(e.GetMsg(),0);
	}

}

std::string ReactionDiffusionSolverFE::toString(){
	return "ReactionDiffusionSolverFE";
}


std::string ReactionDiffusionSolverFE::steerableName(){
	return toString();
}


