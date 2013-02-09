//Maciej Swat

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
#include <PublicUtilities/ParalellUtilsOpenMP.h>

#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
//#define NUMBER_OF_THREADS 4

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// std::ostream & operator<<(std::ostream & out,CompuCell3D::DiffusionData & diffData){
//
//
// }

using namespace CompuCell3D;
using namespace std;


#include "FlexibleDiffusionSolverFE.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverSerializer::serialize(){

	for(int i = 0 ; i < solverPtr->diffSecrFieldTuppleVec.size() ; ++i){
		ostringstream outName;

		outName<<solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName<<"_"<<currentStep<<"."<<serializedFileExtension;
		ofstream outStream(outName.str().c_str());
		solverPtr->outputField( outStream,solverPtr->concentrationFieldVector[i]);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverSerializer::readFromFile(){
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
FlexibleDiffusionSolverFE::FlexibleDiffusionSolverFE()
: DiffusableVectorContiguous<float>(),deltaX(1.0),deltaT(1.0)
{
	serializerPtr=0;
	serializeFlag=false;
	readFromFileFlag=false;
	haveCouplingTerms=false;
	serializeFrequency=0;
	boxWatcherSteppable=0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FlexibleDiffusionSolverFE::~FlexibleDiffusionSolverFE()
{
	if(serializerPtr)
		delete serializerPtr ; serializerPtr=0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverFE::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

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

	cerr<<"INSIDE INIT"<<endl;

	///setting member function pointers
	diffusePtr=&FlexibleDiffusionSolverFE::diffuse;
	secretePtr=&FlexibleDiffusionSolverFE::secrete;

	update(_xmlData,true);

	numberOfFields=diffSecrFieldTuppleVec.size();

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

	//setting up couplingData - field-field interaction terms
	vector<CouplingData>::iterator pos;

	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		pos=diffSecrFieldTuppleVec[i].diffData.couplingDataVec.begin();
		for(int j = 0 ; j < diffSecrFieldTuppleVec[i].diffData.couplingDataVec.size() ; ++j){

			for(int idx=0; idx<concentrationFieldNameVectorTmp.size() ; ++idx){
				if( concentrationFieldNameVectorTmp[idx] == diffSecrFieldTuppleVec[i].diffData.couplingDataVec[j].intrFieldName ){
					diffSecrFieldTuppleVec[i].diffData.couplingDataVec[j].fieldIdx=idx;
					haveCouplingTerms=true; //if this is called at list once we have already coupling terms and need to proceed differently with scratch field initialization
					break;
				}
				//this means that required interacting field name has not been found
				if( idx == concentrationFieldNameVectorTmp.size()-1 ){
					//remove this interacting term
					//                pos=&(diffDataVec[i].degradationDataVec[j]);
					diffSecrFieldTuppleVec[i].diffData.couplingDataVec.erase(pos);
				}
			}
			++pos;
		}
	}

	cerr<<"FIELDS THAT I HAVE"<<endl;
	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		cerr<<"Field "<<i<<" name: "<<concentrationFieldNameVectorTmp[i]<<endl;
	}

	cerr<<"FlexibleDiffusionSolverFE: extra Init in read XML"<<endl;


	///allocate fields including scrartch field
	allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size(),fieldDim); 
	workFieldDim=concentrationFieldVector[0]->getInternalDim();

	//if(!haveCouplingTerms){
	//	allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size()+1,workFieldDim); //+1 is for additional scratch field
	//}else{
	//	allocateDiffusableFieldVector(2*diffSecrFieldTuppleVec.size(),workFieldDim); //with coupling terms every field need to have its own scratch field
	//}

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
void FlexibleDiffusionSolverFE::extraInit(Simulator *simulator){

	if((serializeFlag || readFromFileFlag) && !serializerPtr){
		serializerPtr=new FlexibleDiffusionSolverSerializer();
		serializerPtr->solverPtr=this;
	}

	if(serializeFlag){
		simulator->registerSerializer(serializerPtr);
	}

	bool useBoxWatcher=false;
	for (int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		if(diffSecrFieldTuppleVec[i].diffData.useBoxWatcher){
			useBoxWatcher=true;
			break;
		}
	}
	bool steppableAlreadyRegisteredFlag;
	if(useBoxWatcher){
		boxWatcherSteppable=(BoxWatcher*)Simulator::steppableManager.get("BoxWatcher",&steppableAlreadyRegisteredFlag);
		if(!steppableAlreadyRegisteredFlag)
			boxWatcherSteppable->init(simulator);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverFE::start() {
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

void FlexibleDiffusionSolverFE::initializeConcentration()
{
	for(unsigned int i = 0 ; i <diffSecrFieldTuppleVec.size() ; ++i)
	{
		if(diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;
		cerr << "fail-safe initialization " << diffSecrFieldTuppleVec[i].diffData.concentrationFileName << endl;
		readConcentrationField(diffSecrFieldTuppleVec[i].diffData.concentrationFileName,concentrationFieldVector[i]);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverFE::step(const unsigned int _currentStep) {

	currentStep=_currentStep;

	(this->*secretePtr)();

	(this->*diffusePtr)();

	if(serializeFrequency>0 && serializeFlag && !(_currentStep % serializeFrequency)){
		serializerPtr->setCurrentStep(currentStep);
		serializerPtr->serialize();
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FlexibleDiffusionSolverFE::secreteOnContactSingleField(unsigned int idx){

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
void FlexibleDiffusionSolverFE::secreteSingleField(unsigned int idx){

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

	//ConcentrationField_t & concentrationField=*concentrationFieldVector[idx];

	ConcentrationField_t & concentrationField= *concentrationFieldVector[idx];
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

	for (int z = z_min; z < z_max; z++)
		for (int y = y_min; y < y_max; y++)
			for (int x = x_min; x < x_max; x++){

				pt=Point3D(x-1,y-1,z-1);
				//             cerr<<"pt="<<pt<<" is valid "<<cellFieldG->isValid(pt)<<endl;
				///**
				currentCellPtr=cellFieldG->get(pt);
				//             currentCellPtr=cellFieldG->get(pt);
				//             cerr<<"THIS IS PTR="<<currentCellPtr<<endl;

				//             if(currentCellPtr)
				//                cerr<<"This is id="<<currentCellPtr->id<<endl;
				//currentConcentration = concentrationField.getDirect(x,y,z);

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
						if(currentConcentration*relativeUptakeRateInMedium>maxUptakeInMedium){
							concentrationField.setDirect(x,y,z,currentConcentration-maxUptakeInMedium);
						}else{
							concentrationField.setDirect(x,y,z,currentConcentration - currentConcentration*relativeUptakeRateInMedium);
						}
					}
					if(currentCellPtr){

						mitrUptake=secrData.typeIdUptakeDataMap.find(currentCellPtr->type);
						if(mitrUptake!=end_mitrUptake){
							if(currentConcentration*mitrUptake->second.relativeUptakeRate > mitrUptake->second.maxUptake){
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
void FlexibleDiffusionSolverFE::secreteConstantConcentrationSingleField(unsigned int idx){

	SecretionData & secrData=diffSecrFieldTuppleVec[idx].secrData;

	std::map<unsigned char,float>::iterator mitr;
	std::map<unsigned char,float>::iterator end_mitr=secrData.typeIdSecrConstConstantConcentrationMap.end();

	CellG *currentCellPtr;
	//Field3DImpl<float> * concentrationField=concentrationFieldVector[idx];
	float currentConcentration;
	float secrConst;
	float secrConstMedium=0.0;

	ConcentrationField_t & concentrationField = *concentrationFieldVector[idx];
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

	for (int z = z_min; z < z_max; z++)
		for (int y = y_min; y < y_max; y++)
			for (int x = x_min; x < x_max; x++){

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
void FlexibleDiffusionSolverFE::secrete() {

	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i ){

		for(unsigned int j = 0 ; j <diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size() ; ++j){
			(this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

			//          (this->*secrDataVec[i].secretionFcnPtrVec[j])(i);
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float FlexibleDiffusionSolverFE::couplingTerm(Point3D & _pt,std::vector<CouplingData> & _couplDataVec,float _currentConcentration){

	float couplingTerm=0.0;
	float coupledConcentration;
	for(int i =  0 ; i < _couplDataVec.size() ; ++i){
		coupledConcentration=concentrationFieldVector[_couplDataVec[i].fieldIdx]->get(_pt);
		couplingTerm+=_couplDataVec[i].couplingCoef*_currentConcentration*coupledConcentration;
	}

	return couplingTerm;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverFE::boundaryConditionInit(ConcentrationField_t *concentrationField){
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

void FlexibleDiffusionSolverFE::diffuseSingleField(unsigned int idx){

	// OPTIMIZATIONS - Maciej Swat
	// In addition to using contiguous array with scratch area being interlaced with concentration vector further optimizations are possible
	// In the most innner loop iof the FE solver one can replace maxNeighborIndex with hard coded number. Also instead of 
	// Using boundary strategy to get offset array it is best to hard code offsets and access them directly
	// The downside is that in such a case one woudl have to write separate diffuseSingleField functions fdor 2D, 3D and for hex and square lattices. 
	// However speedups may be worth extra effort.   




	/// 'n' denotes neighbor

	///this is the diffusion equation
	///C_{xx}+C_{yy}+C_{zz}=(1/a)*C_{t}
	///a - diffusivity - diffConst

	///Finite difference method:
	///T_{0,\delta \tau}=F*\sum_{i=1}^N T_{i}+(1-N*F)*T_{0}
	///N - number of neighbors
	///will have to double check this formula

	Point3D pt, n;
	unsigned int token = 0;
	double distance;
	CellG * currentCellPtr=0,*nCell=0;

	short currentCellType=0;
	float concentrationSum=0.0;
	register float updatedConcentration=0.0;

	float currentConcentration=0.0;
	short neighborCounter=0;

	DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;
	float diffConst=diffData.diffConst;
	float decayConst=diffData.decayConst;
	float deltaT=diffData.deltaT;
	float deltaX=diffData.deltaX;
	float dt_dx2=deltaT/(deltaX*deltaX);

	std::set<unsigned char>::iterator sitr;
	std::set<unsigned char>::iterator end_sitr=diffData.avoidTypeIdSet.end();
	std::set<unsigned char>::iterator end_sitr_decay=diffData.avoidDecayInIdSet.end();

	Automaton *automaton=potts->getAutomaton();

	ConcentrationField_t & concentrationField = *concentrationFieldVector[idx];
	ConcentrationField_t * concentrationFieldPtr = concentrationFieldVector[idx];
	//ConcentrationField_t * scratchFieldPtr;

	//if(!haveCouplingTerms)
	//	scratchFieldPtr=concentrationFieldVector[diffSecrFieldTuppleVec.size()];
	//else
	//	scratchFieldPtr=concentrationFieldVector[diffSecrFieldTuppleVec.size()+idx];

	//Array3D_t & concentrationArray = concentrationFieldPtr->getContainer();
	//Array3D_t & scratchArray = scratchFieldPtr->getContainer();

	boundaryConditionInit(concentrationFieldPtr);//initializing boundary conditions

	bool avoidMedium=false;
	bool avoidDecayInMedium=false;
	//the assumption is that medium has type ID 0
	if(diffData.avoidTypeIdSet.find(automaton->getTypeId("Medium")) != end_sitr){
		avoidMedium=true;
	}

	if(diffData.avoidDecayInIdSet.find(automaton->getTypeId("Medium")) != end_sitr_decay){
		avoidDecayInMedium=true;
	}

	unsigned x_min=1,x_max=fieldDim.x+1;
	unsigned y_min=1,y_max=fieldDim.y+1;
	unsigned z_min=1,z_max=fieldDim.z+1;

	cerr<<"x_min="<<x_min<<" x_max="<<x_max<<" y_min="<<y_min<<" y_max="<<y_max<<" z_min="<<z_min<<" z_max="<<z_max<<endl;
	if(diffData.useBoxWatcher){
		Point3D minCoordinates=*(boxWatcherSteppable->getMinCoordinatesPtr());
		Point3D maxCoordinates=*(boxWatcherSteppable->getMaxCoordinatesPtr());
		//cerr<<"FLEXIBLE DIFF SOLVER maxCoordinates="<<maxCoordinates<<" minCoordinates="<<minCoordinates<<endl;
		x_min=minCoordinates.x+1;
		x_max=maxCoordinates.x+1;
		y_min=minCoordinates.y+1;
		y_max=maxCoordinates.y+1;
		z_min=minCoordinates.z+1;
		z_max=maxCoordinates.z+1;
	}
	//cerr<<"x_min="<<x_min<<" x_max="<<x_max<<"y_min="<<y_min<<" y_max="<<y_max<<"z_min="<<z_min<<" z_max="<<z_max<<endl;

	//cerr<<"shiftArray="<<concentrationField.getShiftArray()<<" shiftSwap="<<concentrationField.getShiftSwap()<<endl;
	//har coded offsets for 3D square lattice
	//Point3D offsetArray[6];
	//offsetArray[0]=Point3D(0,0,1);
	//offsetArray[1]=Point3D(0,1,0);
	//offsetArray[2]=Point3D(1,0,0);
	//offsetArray[3]=Point3D(0,0,-1);
	//offsetArray[4]=Point3D(0,-1,0);
	//offsetArray[5]=Point3D(-1,0,0);

	//int counterVec[2];
	//counterVec[0]=0;
	//counterVec[1]=0;


//#pragma omp parallel num_threads(NUMBER_OF_THREADS)
ParallelUtilsOpenMP *pUtils = simulator->getParallelUtils();
cerr<<"numProcs="<<pUtils ->getNumberOfProcessors()<<" number of workNodes="<<pUtils->getNumberOfWorkNodesFESolver()<<endl;

for (int i  =  0 ; i <pUtils->getNumberOfWorkNodesFESolver() ; ++i){
	cerr<<"thread "<<i<<" dimMin="<<pUtils->getFESolverPartition(i).first<<" dimMax="<<pUtils->getFESolverPartition(i).second<<endl;
}

//managing number of threads has to be done BEFORE parallel section otherwise undefined behavior will occur
omp_set_dynamic(0);
omp_set_num_threads(pUtils->getNumberOfWorkNodesFESolver());

#pragma omp parallel
	{	


		//////int x = x_min - 1;
		//////int y = y_min;
		//////int z = z_min;

		//////int xdiff = x_max-x_min;
		//////int ydiff = y_max-y_min;
		//////int zdiff = z_max-z_min;
		//////int workSize = xdiff*ydiff*zdiff;
		CellG *currentCellPtr=0;
		float currentConcentration=0;
		float updatedConcentration=0.0;
		float concentrationSum=0.0;
		short neighborCounter=0;
		CellG *neighborCellPtr=0;




		//#pragma omp for

		int threadNumber=omp_get_thread_num();

		//int workLoad=workSize/omp_get_num_threads();

		//int min_iii=1+workLoad*threadNumber;
		//int max_iii=workLoad*(threadNumber+1);

		//int zL=(z_max-z_min)/omp_get_num_threads();
		//int zMIN=z_min+zL*threadNumber;
		//int zMAX=z_min+zL*(threadNumber+1);
		

		Dim3D minDim=pUtils->getFESolverPartition(threadNumber).first;
		Dim3D maxDim=pUtils->getFESolverPartition(threadNumber).second;

//#pragma omp critical
//		{
//			cerr<<"thread "<<threadNumber<<" dimMin="<<minDim<<" maxDim="<<maxDim<<endl;
//			cerr<<"omp_get_num_threads()="<<omp_get_num_threads()<<endl;
//		}


//#pragma omp critical
//			//cerr<<"threadNumber="<<threadNumber<<" min_iii="<<min_iii<<" max_iii="<<max_iii<<" wL="<<workLoad<<endl;
//			cerr<<"threadNumber="<<threadNumber<<" zMIN="<<zMIN<<" zMAX="<<zMAX<<" zL="<<zL<<" z_min="<<z_min<<" z_max="<<z_max<<endl;
//#pragma omp barrier

		//////for (int z = z_min; z < z_max; z++)
		//////	for (int y = y_min; y < y_max; y++)
		//////		for (int x = x_min; x < x_max; x++){


 
//////#pragma omp for	
//////			   for (int iii=1; iii<=workSize; iii++) {
//////				 x += 1;
//////				 if (x == x_max) {
//////				   x = x_min;
//////				   y += 1;
//////		//	       cerr << "y="<<y<<", x="<<x<<", z="<<z<<endl;
//////				   if (y == y_max) {
//////					 y = y_min;
//////					 z += 1;
//////		//	         cerr << "    z = "<<z<<endl;
//////				   }
//////				 }
				

		//for (int z = z_min; z < z_max; z++)
		////OLD 
		//////for (int z = zMIN; z < zMAX; z++)
		//////	for (int y = y_min; y < y_max; y++)
		//////		for (int x = x_min; x < x_max; x++){

		////NEW
		for (int z = minDim.z; z < maxDim.z; z++)
			for (int y = minDim.y; y < maxDim.y; y++)
				for (int x = minDim.x; x < maxDim.x; x++){
					currentConcentration = concentrationField.getDirect(x,y,z);
					//counterVec[threadNumber]++;

					//#pragma omp critical

//					if (x==6 && y==6  && z==6){
////#pragma omp barrier
////#pragma omp critical
//						cerr<<"PULSE ="<<currentConcentration<<endl;
//					}
//
//					if (x>=5 && x<=7 && y>=5 && y<=7 && z>=5 && z<=7){
//#pragma omp barrier
//#pragma omp critical
//						cerr << "y="<<y<<", x="<<x<<", z="<<z<<" currentConcentration="<<currentConcentration<<" n="<<threadNumber<<endl;
//					}
					//#pragma omp barrier

					//#pragma omp critical
					//		cerr << "y="<<y<<", x="<<x<<", z="<<z<<" currentConcentration="<<currentConcentration<<endl;
					//#pragma omp barrier


					Point3D pt=Point3D(x-1,y-1,z-1);
					currentCellPtr=cellFieldG->getQuick(pt);

					if(avoidMedium && !currentCellPtr){//if medium is to be avoided
						if(avoidDecayInMedium){
							concentrationField.setDirectSwap(x,y,z,currentConcentration);
						}
						else{
							concentrationField.setDirectSwap(x,y,z,currentConcentration-deltaT*(decayConst*currentConcentration));
						}
						continue;
					}

					if(currentCellPtr && diffData.avoidTypeIdSet.find(currentCellPtr->type)!=end_sitr){
						if(diffData.avoidDecayInIdSet.find(currentCellPtr->type)!=end_sitr_decay){
							concentrationField.setDirectSwap(x,y,z,currentConcentration);
						}else{
							concentrationField.setDirectSwap(x,y,z,currentConcentration-deltaT*(decayConst*currentConcentration));
						}
						continue; // avoid user defined types
					}

					updatedConcentration=0.0;
					concentrationSum=0.0;
					neighborCounter=0;

					//loop over nearest neighbors
					CellG *neighborCellPtr=0;


					const std::vector<Point3D> & offsetVecRef=boundaryStrategy->getOffsetVec(pt);
					for (register int i = 0  ; i<=maxNeighborIndex /*offsetVec.size()*/ ; ++i ){
						const Point3D & offset = offsetVecRef[i];

						if(diffData.avoidTypeIdSet.size()||avoidMedium){ //means user defined types to avoid in terms of the diffusion
							n=pt+offset;
							neighborCellPtr=cellFieldG->getQuick(n);
							if(avoidMedium && !neighborCellPtr) continue; // avoid medium if specified by the user
							if(neighborCellPtr && diffData.avoidTypeIdSet.find(neighborCellPtr->type)!=end_sitr) continue;//avoid user specified types
						}
						concentrationSum += concentrationField.getDirect(x+offset.x,y+offset.y,z+offset.z);

						++neighborCounter;
					}




					updatedConcentration =  dt_dx2*diffConst*(concentrationSum - neighborCounter*currentConcentration)+currentConcentration;


					//processing decay depandent on type of the current cell
					if(currentCellPtr){
						if(diffData.avoidDecayInIdSet.find(currentCellPtr->type)!=end_sitr_decay){
							;//decay in this type is forbidden
						}else{
							updatedConcentration-=deltaT*(decayConst*currentConcentration);//decay in this type is allowed
						}
					}else{
						if(avoidDecayInMedium){
							;//decay in Medium is forbidden
						}else{
							updatedConcentration-=deltaT*(decayConst*currentConcentration); //decay in Medium is allowed
						}
					}
					if(haveCouplingTerms){
						updatedConcentration+=couplingTerm(pt,diffData.couplingDataVec,currentConcentration);
					}

					//imposing artificial limits on allowed concentration
					if(diffData.useThresholds){
						if(updatedConcentration>diffData.maxConcentration){
							updatedConcentration=diffData.maxConcentration;
						}
						if(updatedConcentration<diffData.minConcentration){
							updatedConcentration=diffData.minConcentration;
						}
					}


					concentrationField.setDirectSwap(x,y,z,updatedConcentration);//updating scratch

				}

	}

	concentrationField.swapArrays();
	//cerr<<"counterVec[0]="<<counterVec[0]<<" counterVec[1]="<<counterVec[1]<<endl;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool FlexibleDiffusionSolverFE::isBoudaryRegion(int x, int y, int z, Dim3D dim)
{
	if (x < 2 || x > dim.x - 3 || y < 2 || y > dim.y - 3 || z < 2 || z > dim.z - 3 )
		return true;
	else
		return false;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverFE::diffuse() {

	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i ){
		diffuseSingleField(i);
		//if(!haveCouplingTerms){ //without coupling terms field swap takes place immediately aftera given field has been diffused
		//	ConcentrationField_t & concentrationField = *concentrationFieldVector[i];
		//	//ConcentrationField_t * scratchField=concentrationFieldVector[diffSecrFieldTuppleVec.size()];
		//	//copy updated values from scratch to concentration fiel
		//	//scrarch2Concentration(scratchField,concentrationField);
		//	concentrationField.swapArrays();
		//}
	}

	//if(haveCouplingTerms){  //with coupling terms we swap scratch and concentration field after all fields have been diffused
	//	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i ){
	//		ConcentrationField_t * concentrationField=concentrationFieldVector[i];
	//		ConcentrationField_t * scratchField=concentrationFieldVector[diffSecrFieldTuppleVec.size()+i];
	//		scrarch2Concentration(scratchField,concentrationField);
	//	}
	//}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverFE::scrarch2Concentration( ConcentrationField_t *scratchField, ConcentrationField_t *concentrationField){
	//scratchField->switchContainersQuick(*(concentrationField));
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FlexibleDiffusionSolverFE::outputField( std::ostream & _out, ConcentrationField_t *_concentrationField){
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
void FlexibleDiffusionSolverFE::readConcentrationField(std::string fileName,ConcentrationField_t *concentrationField){

	ifstream in(fileName.c_str());

	ASSERT_OR_THROW(string("Could not open chemical concentration file '") +
		fileName + "'!", in.is_open());

	Point3D pt;
	float c;
	//Zero entire field
	for (pt.z = 0; pt.z < fieldDim.z; pt.z++)
		for (pt.y = 0; pt.y < fieldDim.y; pt.y++)
			for (pt.x = 0; pt.x < fieldDim.x; pt.x++){
				//cerr<<"pt="<<pt<<endl;
				concentrationField->set(pt,0);
			}

			while(!in.eof()){
				in>>pt.x>>pt.y>>pt.z>>c;
				if(!in.fail())
					concentrationField->set(pt,c);
			}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FlexibleDiffusionSolverFE::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	//notice, only basic steering is enabled for PDE solvers - changing diffusion constants, do -not-diffuse to types etc...
	// Coupling coefficients cannot be changed and also there is no way to allocate extra fields while simulation is running

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


	diffSecrFieldTuppleVec.clear();

	CC3DXMLElementList diffFieldXMLVec=_xmlData->getElements("DiffusionField");
	for(unsigned int i = 0 ; i < diffFieldXMLVec.size() ; ++i ){
		diffSecrFieldTuppleVec.push_back(DiffusionSecretionFlexFieldTupple());
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
				diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&FlexibleDiffusionSolverFE::secreteSingleField;
				++j;
			}
			else if((*sitr)=="SecretionOnContact"){
				diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&FlexibleDiffusionSolverFE::secreteOnContactSingleField;
				++j;
			}
			else if((*sitr)=="ConstantConcentration"){
				diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&FlexibleDiffusionSolverFE::secreteConstantConcentrationSingleField;
				++j;
			}
		}
	}
}

std::string FlexibleDiffusionSolverFE::toString(){
	return "FlexibleDiffusionSolverFE";
}


std::string FlexibleDiffusionSolverFE::steerableName(){
	return toString();
}
