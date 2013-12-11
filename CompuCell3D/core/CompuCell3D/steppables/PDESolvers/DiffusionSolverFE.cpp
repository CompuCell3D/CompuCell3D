#include "DiffusionSolverFE.h"cell
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
#include <CompuCell3D/plugins/CellTypeMonitor/CellTypeMonitorPlugin.h>

#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <PublicUtilities/StringUtils.h>
#include <PublicUtilities/Vector3.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>

#include "DiffusionSolverFE_CPU.h"
#include "DiffusionSolverFE_CPU_Implicit.h"
#include "GPUEnabled.h"

#include "MyTime.h"

#if OPENCL_ENABLED == 1
#include "OpenCL/DiffusionSolverFE_OpenCL.h"
//#include "OpenCL/DiffusionSolverFE_OpenCL_Implicit.h"
#include "OpenCL/ReactionDiffusionSolverFE_OpenCL_Implicit.h"
#endif

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


#include "DiffusionSolverFE.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverSerializer<Cruncher>::serialize(){

	for(size_t i = 0 ; i < solverPtr->diffSecrFieldTuppleVec.size() ; ++i){
		ostringstream outName;

		outName<<solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName<<"_"<<currentStep<<"."<<serializedFileExtension;
		ofstream outStream(outName.str().c_str());
		solverPtr->outputField( outStream,
			static_cast<Cruncher *>(solverPtr)->getConcentrationField(i));

	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverSerializer<Cruncher>::readFromFile(){
	try{
		for(size_t i = 0 ; i < solverPtr->diffSecrFieldTuppleVec.size() ; ++i){
			ostringstream inName;
			inName<<solverPtr->diffSecrFieldTuppleVec[i].diffData.fieldName<<"."<<serializedFileExtension;

			solverPtr->readConcentrationField(inName.str().c_str(),
				static_cast<Cruncher *>(solverPtr)->getConcentrationField(i));
		}
	} catch (BasicException &e) {
		cerr<<"COULD NOT FIND ONE OF THE FILES"<<endl;
		throw BasicException("Error in reading diffusion fields from file",e);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
DiffusionSolverFE<Cruncher>::DiffusionSolverFE()
:deltaX(1.0),deltaT(1.0), latticeType(SQUARE_LATTICE)
{
	serializerPtr=0;
	pUtils=0;
	serializeFlag=false;
	readFromFileFlag=false;
	haveCouplingTerms=false;
	serializeFrequency=0;
	boxWatcherSteppable=0;
	diffusionLatticeScalingFactor=1.0;
	autoscaleDiffusion=false;
	cellTypeMonitorPlugin=0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
DiffusionSolverFE<Cruncher>::~DiffusionSolverFE()
{
	if(serializerPtr){
		delete serializerPtr ; 
		serializerPtr=0;
	}
}


template <class Cruncher>
void DiffusionSolverFE<Cruncher>::Scale(std::vector<float> const &maxDiffConstVec, float maxStableDiffConstant)
{
	//scaling of diffusion and secretion coeeficients
	for(unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); i++){
		scalingExtraMCSVec[i] = ceil(maxDiffConstVec[i]/maxStableDiffConstant); //compute number of calls to diffusion solver
		if (scalingExtraMCSVec[i]==0)
			continue;

		//diffusion data
		for(int currentCellType = 0; currentCellType < UCHAR_MAX+1; currentCellType++) {
			float diffConstTemp = diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType];					
			float decayConstTemp = diffSecrFieldTuppleVec[i].diffData.decayCoef[currentCellType];
			diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType] = (diffConstTemp/scalingExtraMCSVec[i]); //scale diffusion
			diffSecrFieldTuppleVec[i].diffData.decayCoef[currentCellType] = (decayConstTemp/scalingExtraMCSVec[i]); //scale decay
			
		}
	
		//secretion data
		SecretionData & secrData=diffSecrFieldTuppleVec[i].secrData;
		for (std::map<unsigned char,float>::iterator mitr=secrData.typeIdSecrConstMap.begin() ; mitr!=secrData.typeIdSecrConstMap.end() ; ++mitr){
			mitr->second/=scalingExtraMCSVec[i];
		}

		for (std::map<unsigned char,float>::iterator mitr=secrData.typeIdSecrConstConstantConcentrationMap.begin() ; mitr!=secrData.typeIdSecrConstConstantConcentrationMap.end() ; ++mitr){
			mitr->second/=scalingExtraMCSVec[i];
		}

		for (std::map<unsigned char,SecretionOnContactData>::iterator mitr=secrData.typeIdSecrOnContactDataMap.begin() ; mitr!=secrData.typeIdSecrOnContactDataMap.end() ; ++mitr){
			SecretionOnContactData & secrOnContactData=mitr->second;
			for (std::map<unsigned char,float>::iterator cmitr=secrOnContactData.contactCellMap.begin() ; cmitr!=secrOnContactData.contactCellMap.end() ; ++cmitr){
				cmitr->second/=scalingExtraMCSVec[i];
			}	
		}

		//uptake data
		for (std::map<unsigned char,UptakeData>::iterator mitr=secrData.typeIdUptakeDataMap.begin() ; mitr!=secrData.typeIdUptakeDataMap.end() ; ++mitr){
			mitr->second.maxUptake/=scalingExtraMCSVec[i];
			mitr->second.relativeUptakeRate/=scalingExtraMCSVec[i];			
		}
	}
}

template <class Cruncher>
bool DiffusionSolverFE<Cruncher>::hasAdditionalTerms()const{

	for(size_t i=0; i<diffSecrFieldTuppleVec.size(); ++i){
		DiffusionData const &diffData=diffSecrFieldTuppleVec[i].diffData;
		if(!diffData.additionalTerm.empty())
			return true;
	}
	return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

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
	diffusePtr=&DiffusionSolverFE::diffuse;
	secretePtr=&DiffusionSolverFE::secrete;

	update(_xmlData,true);

	cerr<<"AFTER UPDATE"<<endl;

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

	float maxDiffConst = 0.0;
	scalingExtraMCS = 0;
	std::vector<float> maxDiffConstVec; 

	scalingExtraMCSVec.assign(diffSecrFieldTuppleVec.size(),0);
	maxDiffConstVec.assign(diffSecrFieldTuppleVec.size(),0.0);

	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		pos=diffSecrFieldTuppleVec[i].diffData.couplingDataVec.begin();
		for(size_t j = 0 ; j < diffSecrFieldTuppleVec[i].diffData.couplingDataVec.size() ; ++j){

			for(size_t idx=0; idx<concentrationFieldNameVectorTmp.size() ; ++idx){
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

		for(int currentCellType = 0; currentCellType < UCHAR_MAX+1; currentCellType++) {
			//                 cout << "diffCoef[currentCellType]: " << diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType] << endl;
			maxDiffConstVec[i] = (maxDiffConstVec[i] < diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType]) ? diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType]: maxDiffConstVec[i];
		}
	}


	cerr<<"FIELDS THAT I HAVE"<<endl;
	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		cerr<<"Field "<<i<<" name: "<<concentrationFieldNameVectorTmp[i]<<endl;
	}

	cerr<<"DiffusionSolverFE: extra Init in read XML"<<endl;

	///allocate fields including scrartch field
	static_cast<Cruncher*>(this)->allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size(),fieldDim); 
	workFieldDim=static_cast<Cruncher*>(this)->getConcentrationField(0)->getInternalDim();

	//if(!haveCouplingTerms){
	//	allocateDiffusableFieldVector(diffSecrFieldTuppleVec.size()+1,workFieldDim); //+1 is for additional scratch field
	//}else{
	//	allocateDiffusableFieldVector(2*diffSecrFieldTuppleVec.size(),workFieldDim); //with coupling terms every field need to have its own scratch field
	//}

	//here I need to copy field names from concentrationFieldNameVectorTmp to concentrationFieldNameVector
	//because concentrationFieldNameVector is reallocated with default values once I call allocateDiffusableFieldVector

	for(unsigned int i=0 ; i < concentrationFieldNameVectorTmp.size() ; ++i){
		static_cast<Cruncher*>(this)->setConcentrationFieldName(i,concentrationFieldNameVectorTmp[i]);
	}

	//register fields once they have been allocated
	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
		simPtr->registerConcentrationField(
			static_cast<Cruncher*>(this)->getConcentrationFieldName(i), 
			static_cast<Cruncher*>(this)->getConcentrationField(i));
		cerr<<"registring field: "<<
			static_cast<Cruncher*>(this)->getConcentrationFieldName(i)<<" field address="<<
			static_cast<Cruncher*>(this)->getConcentrationField(i)<<endl;
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

	float extraCheck;

	// //check diffusion constant and scale extraTimesPerMCS

	float maxStableDiffConstant=0.23;
	if(static_cast<Cruncher*>(this)->getBoundaryStrategy()->getLatticeType()==HEXAGONAL_LATTICE) {
		if (fieldDim.x==1 || fieldDim.y==1||fieldDim.z==1){ //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D		
			maxStableDiffConstant=0.16f;
		}else{//3D
			maxStableDiffConstant=0.08f;
		}
	}else{//Square lattice
		if (fieldDim.x==1 || fieldDim.y==1||fieldDim.z==1){ //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D				
			maxStableDiffConstant=0.23f;
		}else{//3D
			maxStableDiffConstant=0.14f;
		}
	}

	Scale(maxDiffConstVec, maxStableDiffConstant);//TODO: remove for implicit solvers?

	//determining latticeType and setting diffusionLatticeScalingFactor
	//When you evaluate div as a flux through the surface divided bby volume those scaling factors appear automatically. On cartesian lattife everythink is one so this is easy to forget that on different lattices they are not1
	diffusionLatticeScalingFactor=1.0;
	if (static_cast<Cruncher*>(this)->getBoundaryStrategy()->getLatticeType()==HEXAGONAL_LATTICE){
		if (fieldDim.x==1 || fieldDim.y==1||fieldDim.z==1){ //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D
			diffusionLatticeScalingFactor=1.0f/sqrt(3.0f);// (2/3)/dL^2 dL=sqrt(2/sqrt(3)) so (2/3)/dL^2=1/sqrt(3)
		}else{//3D simulation
			diffusionLatticeScalingFactor=pow(2.0f,-4.0f/3.0f); //(1/2)/dL^2 dL dL^2=2**(1/3) so (1/2)/dL^2=1/(2.0*2^(1/3))=2^(-4/3)
		}

	}
	//this is no longer the case - we kept this option form backward compatibility reasons for flexibleDiffusion solver
	////we only autoscale diffusion when user requests it explicitely
	//if (!autoscaleDiffusion){
	//	diffusionLatticeScalingFactor=1.0;
	//}


	bool pluginAlreadyRegisteredFlag;
	cellTypeMonitorPlugin=(CellTypeMonitorPlugin*)Simulator::pluginManager.get("CellTypeMonitor",&pluginAlreadyRegisteredFlag);
	if(!pluginAlreadyRegisteredFlag){
		cellTypeMonitorPlugin->init(simulator);	
		h_celltype_field=cellTypeMonitorPlugin->getCellTypeArray();

	}


	simulator->registerSteerableObject(this);

	//platform-specific initialization
	initImpl();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::extraInit(Simulator *simulator){

	if((serializeFlag || readFromFileFlag) && !serializerPtr){
		serializerPtr=new DiffusionSolverSerializer<Cruncher>();
		serializerPtr->solverPtr=this;
	}

	if(serializeFlag){
		simulator->registerSerializer(serializerPtr);
	}

	bool useBoxWatcher=false;
	for (size_t i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
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

	prepareForwardDerivativeOffsets();

	//platform-specific initialization
	extraInitImpl();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void  DiffusionSolverFE<Cruncher>::handleEvent(CC3DEvent & _event){

	if (_event.id!=LATTICE_RESIZE){
		return;
	}

    
    
	static_cast<Cruncher *>(this)->handleEventLocal(_event);
    
	h_celltype_field=cellTypeMonitorPlugin->getCellTypeArray();

	fieldDim=cellFieldG->getDim();
    workFieldDim=static_cast<Cruncher*>(this)->getConcentrationField(0)->getInternalDim();

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
bool DiffusionSolverFE<Cruncher>::checkIfOffsetInArray(Point3D _pt, vector<Point3D> &_array){
	for (size_t i  = 0 ; i<_array.size() ;++i ){
		if(_array[i]==_pt)
			return true;
	}
	return false;
}

template <class Cruncher>
void DiffusionSolverFE<Cruncher>::prepareForwardDerivativeOffsets(){
	latticeType=static_cast<Cruncher*>(this)->getBoundaryStrategy()->getLatticeType();


	unsigned int maxHexArraySize=6;

	hexOffsetArray.assign(maxHexArraySize,vector<Point3D>());


	if(latticeType==HEXAGONAL_LATTICE){//2D case

		if (fieldDim.x==1||fieldDim.y==1||fieldDim.z==1){

			hexOffsetArray[0].push_back(Point3D(0,1,0));
			hexOffsetArray[0].push_back(Point3D(1,1,0));
			hexOffsetArray[0].push_back(Point3D(1,0,0));


			hexOffsetArray[1].push_back(Point3D(0,1,0));
			hexOffsetArray[1].push_back(Point3D(-1,1,0));
			hexOffsetArray[1].push_back(Point3D(1,0,0));

			hexOffsetArray[2]=hexOffsetArray[0];
			hexOffsetArray[4]=hexOffsetArray[0];


			hexOffsetArray[3]=hexOffsetArray[1];
			hexOffsetArray[5]=hexOffsetArray[1];




		}else{ //3D case - we assume that forward derivatives are calculated using 3 sides with z=1 and 3 sides which have same offsets os for 2D case (with z=0)
			hexOffsetArray.assign(maxHexArraySize,vector<Point3D>(6));

			//y%2=0 and z%3=0						
			hexOffsetArray[0][0]=Point3D(0,1,0);
			hexOffsetArray[0][1]=Point3D(1,1,0);
			hexOffsetArray[0][2]=Point3D(1,0,0);
			hexOffsetArray[0][3]=Point3D(0,-1,1);
			hexOffsetArray[0][4]=Point3D(0,0,1);
			hexOffsetArray[0][5]=Point3D(1,0,1);



			//y%2=1 and z%3=0						
			hexOffsetArray[1][0]=Point3D(-1,1,0);
			hexOffsetArray[1][1]=Point3D(0,1,0);
			hexOffsetArray[1][2]=Point3D(1,0,0);
			hexOffsetArray[1][3]=Point3D(-1,0,1);
			hexOffsetArray[1][4]=Point3D(0,0,1);
			hexOffsetArray[1][5]=Point3D(0,-1,1);


			//y%2=0 and z%3=1						
			hexOffsetArray[2][0]=Point3D(-1,1,0);
			hexOffsetArray[2][1]=Point3D(0,1,0);
			hexOffsetArray[2][2]=Point3D(1,0,0);
			hexOffsetArray[2][3]=Point3D(0,1,1);
			hexOffsetArray[2][4]=Point3D(0,0,1);
			hexOffsetArray[2][5]=Point3D(-1,1,1);


			//y%2=1 and z%3=1						
			hexOffsetArray[3][0]=Point3D(0,1,0);
			hexOffsetArray[3][1]=Point3D(1,1,0);
			hexOffsetArray[3][2]=Point3D(1,0,0);
			hexOffsetArray[3][3]=Point3D(0,1,1);			
			hexOffsetArray[3][4]=Point3D(0,0,1);
			hexOffsetArray[3][5]=Point3D(1,1,1);


			//y%2=0 and z%3=2						
			hexOffsetArray[4][0]=Point3D(-1,1,0);
			hexOffsetArray[4][1]=Point3D(0,1,0);
			hexOffsetArray[4][2]=Point3D(1,0,0);;
			hexOffsetArray[4][3]=Point3D(-1,0,1);
			hexOffsetArray[4][4]=Point3D(0,0,1);
			hexOffsetArray[4][5]=Point3D(0,-1,1);


			//y%2=1 and z%3=2						
			hexOffsetArray[5][0]=Point3D(0,1,0);
			hexOffsetArray[5][1]=Point3D(1,1,0);
			hexOffsetArray[5][2]=Point3D(1,0,0);
			hexOffsetArray[5][3]=Point3D(0,-1,1);
			hexOffsetArray[5][4]=Point3D(0,0,1);
			hexOffsetArray[5][5]=Point3D(1,0,1);
		}
	}else{
		Point3D pt(fieldDim.x/2,fieldDim.y/2,fieldDim.z/2); // pick point in the middle of the lattice

		const std::vector<Point3D> & offsetVecRef=static_cast<Cruncher*>(this)->getBoundaryStrategy()->getOffsetVec(pt);
				for (unsigned int i = 0  ; i<=static_cast<Cruncher*>(this)->getMaxNeighborIndex(); ++i ){

			const Point3D & offset = offsetVecRef[i];
			//we use only those offset vectors which have only positive coordinates
			if (offset.x>=0 &&offset.y>=0 &&offset.z>=0){
				offsetVecCartesian.push_back(offset);
			}			
		}		

	}


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::start() {
	//     if(diffConst> (1.0/6.0-0.05) ){ //hard coded condtion for stability of the solutions - assume dt=1 dx=dy=dz=1
	//
	//       cerr<<"CANNOT SOLVE DIFFUSION EQUATION: STABILITY PROBLEM - DIFFUSION CONSTANT TOO LARGE. EXITING..."<<endl;
	//       exit(0);
	//
	//    }

	dt_dx2=deltaT/(deltaX*deltaX);

	if (simPtr->getRestartEnabled()){
		return ;  // we will not initialize cells if restart flag is on
	}

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

	m_RDTime=0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::finish(){
	cerr<<m_RDTime<<" ms spent in solving "<<(hasAdditionalTerms()?"reaction-":"")<<"diffusion problem"<<endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::initializeConcentration()
{
	for(unsigned int i = 0 ; i <diffSecrFieldTuppleVec.size() ; ++i)
	{
		//cerr<<"EXPRESSION TO EVALUATE "<<diffSecrFieldTuppleVec[i].diffData.initialConcentrationExpression<<endl;
		if(!diffSecrFieldTuppleVec[i].diffData.initialConcentrationExpression.empty()){
			static_cast<Cruncher*>(this)->initializeFieldUsingEquation(
				static_cast<Cruncher*>(this)->getConcentrationField(i),
				diffSecrFieldTuppleVec[i].diffData.initialConcentrationExpression);
			continue;
		}
		if(diffSecrFieldTuppleVec[i].diffData.concentrationFileName.empty()) continue;
		cerr << "fail-safe initialization " << diffSecrFieldTuppleVec[i].diffData.concentrationFileName << endl;
		readConcentrationField(diffSecrFieldTuppleVec[i].diffData.concentrationFileName,
			static_cast<Cruncher*>(this)->getConcentrationField(i));
		//cerr<<"---------------------------------Have read something from a file----------------------------------\n";
		//float f=static_cast<Cruncher*>(this)->getConcentrationField(i)->getDirect(20,20,32);
	
		//cerr<<"\tcontrol value is"<<f<<endl;

	}
}

template <class Cruncher>
void DiffusionSolverFE<Cruncher>::stepImpl(const unsigned int _currentStep)
{
	//cerr<<"diffSecrFieldTuppleVec.size()="<<diffSecrFieldTuppleVec.size()<<endl;
	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i ){
		//cerr<<"scalingExtraMCSVec[i]="<<scalingExtraMCSVec[i]<<endl;
		if (!scalingExtraMCSVec[i]){ //we do not call diffusion step but call secretion
			for(unsigned int j = 0 ; j <diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size() ; ++j){
				(this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

			}
		}

		//cerr<<"Making "<<scalingExtraMCSVec[i]<<" extra diffusion steps"<<endl;
		for(int extraMCS = 0; extraMCS < scalingExtraMCSVec[i]; extraMCS++) {
//			std::cout<<"field #"<<i<<"; diffConst is: "<<diffSecrFieldTuppleVec[i].diffData.diffConst<<"; "<<
//				diffSecrFieldTuppleVec[i].diffData.diffCoef[0]<<"; "<<
//				diffSecrFieldTuppleVec[i].diffData.diffCoef[1]<<std::endl;
			diffuseSingleField(i);
			for(unsigned int j = 0 ; j <diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size() ; ++j){
				(this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::step(const unsigned int _currentStep) {

	currentStep=_currentStep;

	MyTime::Time_t stepBT=MyTime::CTime();
	stepImpl(_currentStep);
	m_RDTime+=MyTime::ElapsedTime(stepBT, MyTime::CTime());
	//std::cout<<"Solving took "<<MyTime::ElapsedTime(stepBT, MyTime::CTime())<<"ms"<<std::endl;
	
	
	if(serializeFrequency>0 && serializeFlag && !(_currentStep % serializeFrequency)){
		serializerPtr->setCurrentStep(currentStep);
		serializerPtr->serialize();
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::getMinMaxBox(bool useBoxWatcher, int threadNumber, Dim3D &minDim, Dim3D &maxDim)const{
	bool isMaxThread;
	if(useBoxWatcher){
		minDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).first;
		maxDim=pUtils->getFESolverPartitionWithBoxWatcher(threadNumber).second;

		isMaxThread=(threadNumber==pUtils->getNumberOfWorkNodesFESolverWithBoxWatcher()-1);

	}else{
		minDim=pUtils->getFESolverPartition(threadNumber).first;
		maxDim=pUtils->getFESolverPartition(threadNumber).second;

		isMaxThread=(threadNumber==pUtils->getNumberOfWorkNodesFESolver()-1);
	}

	if(!hasExtraLayer()){
		if(threadNumber==0){
			minDim-=Dim3D(1,1,1);
		}

		if(isMaxThread){
			maxDim-=Dim3D(1,1,1);
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::secreteOnContactSingleField(unsigned int idx){

	SecretionData & secrData=diffSecrFieldTuppleVec[idx].secrData;

	std::map<unsigned char,SecretionOnContactData>::iterator mitrShared;
	std::map<unsigned char,SecretionOnContactData>::iterator end_mitr=secrData.typeIdSecrOnContactDataMap.end();

	typename Cruncher::ConcentrationField_t & concentrationField= *static_cast<Cruncher *>(this)->getConcentrationField(idx);
	
	std::map<unsigned char, float> * contactCellMapMediumPtr;
	std::map<unsigned char, float> * contactCellMapPtr;


	bool secreteInMedium=false;
	//the assumption is that medium has type ID 0
	mitrShared=secrData.typeIdSecrOnContactDataMap.find(automaton->getTypeId("Medium"));

	if(mitrShared != end_mitr ){
		secreteInMedium=true;
		contactCellMapMediumPtr = &(mitrShared->second.contactCellMap);
	}


	//HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES
	DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;

	if(diffData.useBoxWatcher){

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

	pUtils->prepareParallelRegionFESolvers(diffData.useBoxWatcher);
#pragma omp parallel
	{	

		std::map<unsigned char,SecretionOnContactData>::iterator mitr;
		std::map<unsigned char, float>::iterator mitrTypeConst;

		float currentConcentration;
		float secrConst;
		float secrConstMedium=0.0;

		CellG *currentCellPtr;
		Point3D pt;
		Neighbor n;
		CellG *nCell=0;
		WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
		unsigned char type;

		int threadNumber=pUtils->getCurrentWorkNodeNumber();

		bool hasExtraBndLayer=hasExtraLayer();

		Dim3D minDim;		
		Dim3D maxDim;

		getMinMaxBox(diffData.useBoxWatcher, threadNumber, minDim, maxDim);
		
		for (int z = minDim.z; z < maxDim.z; z++)
			for (int y = minDim.y; y < maxDim.y; y++)
				for (int x = minDim.x; x < maxDim.x; x++){
					if(hasExtraBndLayer)
						pt=Point3D(x-1,y-1,z-1);
					else
						pt=Point3D(x,y,z);
					///**
					currentCellPtr=cellFieldG->getQuick(pt);
					//             currentCellPtr=cellFieldG->get(pt);
					currentConcentration = concentrationField.getDirect(x,y,z);

					if(secreteInMedium && ! currentCellPtr){
						for (unsigned int i = 0  ; i<=static_cast<Cruncher *>(this)->getMaxNeighborIndex(); ++i ){
							n=static_cast<Cruncher *>(this)->getBoundaryStrategy()->getNeighborDirect(pt,i);
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

							for (unsigned int i = 0  ; i<=static_cast<Cruncher *>(this)->getMaxNeighborIndex(); ++i ){

								n=static_cast<Cruncher *>(this)->getBoundaryStrategy()->getNeighborDirect(pt,i);
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
}


//Default implementation. Most of the solvers have it.
template <class Cruncher>
bool DiffusionSolverFE<Cruncher>::hasExtraLayer()const{
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::secreteSingleField(unsigned int idx){

	SecretionData & secrData=diffSecrFieldTuppleVec[idx].secrData;

	float maxUptakeInMedium=0.0;
	float relativeUptakeRateInMedium=0.0;
	float secrConstMedium=0.0;

	std::map<unsigned char,float>::iterator mitrShared;
	std::map<unsigned char,float>::iterator end_mitr=secrData.typeIdSecrConstMap.end();
	std::map<unsigned char,UptakeData>::iterator mitrUptakeShared;
	std::map<unsigned char,UptakeData>::iterator end_mitrUptake=secrData.typeIdUptakeDataMap.end();

	typename Cruncher::ConcentrationField_t &concentrationField= *static_cast<Cruncher *>(this)->getConcentrationField(idx);

	bool doUptakeFlag=false;
	bool uptakeInMediumFlag=false;
	bool secreteInMedium=false;
	//the assumption is that medium has type ID 0
	mitrShared=secrData.typeIdSecrConstMap.find(automaton->getTypeId("Medium"));

	if( mitrShared != end_mitr){
		secreteInMedium=true;
		secrConstMedium=mitrShared->second;
	}

	//uptake for medium setup
	if(secrData.typeIdUptakeDataMap.size()){
		doUptakeFlag=true;
	}
	//uptake for medium setup
	if(doUptakeFlag){
		mitrUptakeShared=secrData.typeIdUptakeDataMap.find(automaton->getTypeId("Medium"));
		if(mitrUptakeShared != end_mitrUptake){
			maxUptakeInMedium=mitrUptakeShared->second.maxUptake;
			relativeUptakeRateInMedium=mitrUptakeShared->second.relativeUptakeRate;
			uptakeInMediumFlag=true;

		}
	}

	//HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES
	DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;
	if(diffData.useBoxWatcher){

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

	//cerr<<"SECRETE SINGLE FIELD"<<endl;


	pUtils->prepareParallelRegionFESolvers(diffData.useBoxWatcher);


#pragma omp parallel
	{	

		CellG *currentCellPtr;
		//Field3DImpl<float> * concentrationField=concentrationFieldVector[idx];
		float currentConcentration;
		float secrConst;


		std::map<unsigned char,float>::iterator mitr;
		std::map<unsigned char,UptakeData>::iterator mitrUptake;

		Point3D pt;
		int threadNumber=pUtils->getCurrentWorkNodeNumber();


		bool hasExtraBndLayer=hasExtraLayer();

		Dim3D minDim;		
		Dim3D maxDim;

		getMinMaxBox(diffData.useBoxWatcher, threadNumber, minDim, maxDim);
		

		for (int z = minDim.z; z < maxDim.z; z++)
			for (int y = minDim.y; y < maxDim.y; y++)
				for (int x = minDim.x; x < maxDim.x; x++){

					if(hasExtraBndLayer)
						pt=Point3D(x-1,y-1,z-1);
					else
						pt=Point3D(x,y,z);
					//             cerr<<"pt="<<pt<<" is valid "<<cellFieldG->isValid(pt)<<endl;
					///**
					currentCellPtr=cellFieldG->getQuick(pt);
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
								concentrationField.setDirect(x,y,z,concentrationField.getDirect(x,y,z)-maxUptakeInMedium);
							}else{
								concentrationField.setDirect(x,y,z,concentrationField.getDirect(x,y,z) - currentConcentration*relativeUptakeRateInMedium);
							}
						}
						if(currentCellPtr){

							mitrUptake=secrData.typeIdUptakeDataMap.find(currentCellPtr->type);
							if(mitrUptake!=end_mitrUptake){								
								if(currentConcentration*mitrUptake->second.relativeUptakeRate > mitrUptake->second.maxUptake){
									concentrationField.setDirect(x,y,z,concentrationField.getDirect(x,y,z)-mitrUptake->second.maxUptake);
									//cerr<<" uptake concentration="<< currentConcentration<<" relativeUptakeRate="<<mitrUptake->second.relativeUptakeRate<<" subtract="<<mitrUptake->second.maxUptake<<endl;
								}else{
									concentrationField.setDirect(x,y,z,concentrationField.getDirect(x,y,z)-currentConcentration*mitrUptake->second.relativeUptakeRate);
									//cerr<<"concentration="<< currentConconcentrationField.getDirect(x,y,z)- currentConcentration*mitrUptake->second.relativeUptakeRate);
								}
							}
						}
					}
				}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::secreteConstantConcentrationSingleField(unsigned int idx){

	// std::cerr<<"***************here secreteConstantConcentrationSingleField***************\n";

	SecretionData & secrData=diffSecrFieldTuppleVec[idx].secrData;

	std::map<unsigned char,float>::iterator mitrShared;
	std::map<unsigned char,float>::iterator end_mitr=secrData.typeIdSecrConstConstantConcentrationMap.end();


	float secrConstMedium=0.0;

	typename Cruncher::ConcentrationField_t & concentrationField = *static_cast<Cruncher *>(this)->getConcentrationField(idx);
	
	bool secreteInMedium=false;
	//the assumption is that medium has type ID 0
	mitrShared=secrData.typeIdSecrConstConstantConcentrationMap.find(automaton->getTypeId("Medium"));

	if( mitrShared != end_mitr){
		secreteInMedium=true;
		secrConstMedium=mitrShared->second;
	}


	//HAVE TO WATCH OUT FOR SHARED/PRIVATE VARIABLES
	DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;
	if(diffData.useBoxWatcher){

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

	pUtils->prepareParallelRegionFESolvers(diffData.useBoxWatcher);

#pragma omp parallel
	{	

		CellG *currentCellPtr;
		//Field3DImpl<float> * concentrationField=concentrationFieldVector[idx];
		float currentConcentration;
		float secrConst;

		std::map<unsigned char,float>::iterator mitr;

		Point3D pt;
		int threadNumber=pUtils->getCurrentWorkNodeNumber();

		bool hasExtraBndLayer=hasExtraLayer();

		Dim3D minDim;		
		Dim3D maxDim;

		getMinMaxBox(diffData.useBoxWatcher, threadNumber, minDim, maxDim);
		
		for (int z = minDim.z; z < maxDim.z; z++)
			for (int y = minDim.y; y < maxDim.y; y++)
				for (int x = minDim.x; x < maxDim.x; x++){

					if(hasExtraBndLayer)
						pt=Point3D(x-1,y-1,z-1);
					else
						pt=Point3D(x,y,z);
					//             cerr<<"pt="<<pt<<" is valid "<<cellFieldG->isValid(pt)<<endl;
					///**
					currentCellPtr=cellFieldG->getQuick(pt);
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
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::secrete() {

	for(unsigned int i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i ){	

		for(unsigned int j = 0 ; j <diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size() ; ++j){
			(this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);

			//          (this->*secrDataVec[i].secretionFcnPtrVec[j])(i);
		}

	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
float DiffusionSolverFE<Cruncher>::couplingTerm(Point3D & _pt,std::vector<CouplingData> & _couplDataVec,float _currentConcentration){

	float couplingTerm=0.0;
	float coupledConcentration;
	for(size_t i =  0 ; i < _couplDataVec.size() ; ++i){
		coupledConcentration=static_cast<Cruncher *>(this)->getConcentrationField(_couplDataVec[i].fieldIdx)->get(_pt);
		couplingTerm+=_couplDataVec[i].couplingCoef*_currentConcentration*coupledConcentration;
	}

	return couplingTerm;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::boundaryConditionInit(int idx){

	typename Cruncher::ConcentrationField_t & _array = *static_cast<Cruncher *>(this)->getConcentrationField(idx);
	bool detailedBCFlag=bcSpecFlagVec[idx];
	BoundaryConditionSpecifier & bcSpec=bcSpecVec[idx];
	DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;
	float deltaX=diffData.deltaX;

	//ConcentrationField_t & _array=*concentrationField;
	if (!detailedBCFlag){
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

	}else{
		//detailed specification of boundary conditions
		// X axis
		if (bcSpec.planePositions[0]==BoundaryConditionSpecifier::PERIODIC || bcSpec.planePositions[1]==BoundaryConditionSpecifier::PERIODIC){
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

		}else{

			if (bcSpec.planePositions[0]==BoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[0];
				int x=0;
				for(int y=0 ; y< workFieldDim.y-1; ++y)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[0]==BoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
				float cdValue= bcSpec.values[0];
				int x=0;

				for(int y=0 ; y< workFieldDim.y-1; ++y)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,_array.getDirect(1,y,z)-cdValue*deltaX);
					}

			}

			if (bcSpec.planePositions[1]==BoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[1];
				int x=fieldDim.x+1;
				for(int y=0 ; y< workFieldDim.y-1; ++y)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[1]==BoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
				float cdValue= bcSpec.values[1];
				int x=fieldDim.x+1;

				for(int y=0 ; y< workFieldDim.y-1; ++y)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,_array.getDirect(x-1,y,z)+cdValue*deltaX);
					}

			}

		}
		//detailed specification of boundary conditions
		// Y axis
		if (bcSpec.planePositions[2]==BoundaryConditionSpecifier::PERIODIC || bcSpec.planePositions[3]==BoundaryConditionSpecifier::PERIODIC){
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

		}else{

			if (bcSpec.planePositions[2]==BoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[2];
				int y=0;
				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[2]==BoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
				float cdValue= bcSpec.values[2];
				int y=0;

				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,_array.getDirect(x,1,z)-cdValue*deltaX);
					}

			}

			if (bcSpec.planePositions[3]==BoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[3];
				int y=fieldDim.y+1;
				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[3]==BoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
				float cdValue= bcSpec.values[3];
				int y=fieldDim.y+1;

				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,_array.getDirect(x,y-1,z)+cdValue*deltaX);
					}
			}

		}
		//detailed specification of boundary conditions
		// Z axis
		if (bcSpec.planePositions[4]==BoundaryConditionSpecifier::PERIODIC || bcSpec.planePositions[5]==BoundaryConditionSpecifier::PERIODIC){
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

		}else{

			if (bcSpec.planePositions[4]==BoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[4];
				int z=0;
				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int y=0 ; y<workFieldDim.y-1 ; ++y){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[4]==BoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
				float cdValue= bcSpec.values[4];
				int z=0;

				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int y=0 ; y<workFieldDim.y-1 ; ++y){
						_array.setDirect(x,y,z,_array.getDirect(x,y,1)-cdValue*deltaX);
					}

			}

			if (bcSpec.planePositions[5]==BoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[5];
				int z=fieldDim.z+1;
				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int y=0 ; y<workFieldDim.y-1 ; ++y){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[5]==BoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
				float cdValue= bcSpec.values[5];
				int z=fieldDim.z+1;

				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int y=0 ; y<workFieldDim.y-1 ; ++y){
						_array.setDirect(x,y,z,_array.getDirect(x,y,z-1)+cdValue*deltaX);
					}
			}

		}

	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::diffuseSingleField(unsigned int idx)
{

	boundaryConditionInit(idx);//initializing boundary conditions

	DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;
	typename Cruncher::ConcentrationField_t & concentrationField = *static_cast<Cruncher *>(this)->getConcentrationField(idx);

	initCellTypesAndBoundariesImpl();
	static_cast<Cruncher *>(this)->diffuseSingleFieldImpl(concentrationField, diffData);
	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
bool DiffusionSolverFE<Cruncher>::isBoudaryRegion(int x, int y, int z, Dim3D dim)
{
	if (x < 2 || x > dim.x - 3 || y < 2 || y > dim.y - 3 || z < 2 || z > dim.z - 3 )
		return true;
	else
		return false;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::diffuse() {


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
template <class ConcentrationField_t>
void DiffusionSolverFE<Cruncher>::scrarch2Concentration( ConcentrationField_t *scratchField, ConcentrationField_t *concentrationField){
	//scratchField->switchContainersQuick(*(concentrationField));
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Cruncher>
template <class ConcentrationField_t>
void DiffusionSolverFE<Cruncher>::outputField( std::ostream & _out, ConcentrationField_t *_concentrationField){
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
template <class Cruncher>
template <class ConcentrationField_t>
void DiffusionSolverFE<Cruncher>::readConcentrationField(std::string fileName,ConcentrationField_t *concentrationField){

	std::string basePath=simulator->getBasePath();
	std::string fn=fileName;
	if (basePath!=""){
		fn	= basePath+"/"+fileName;
	}

	ifstream in(fn.c_str());

	ASSERT_OR_THROW(string("Could not open chemical concentration file '") +
		fn	 + "'!", in.is_open());

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
template <class Cruncher>
void DiffusionSolverFE<Cruncher>::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	//notice, only basic steering is enabled for PDE solvers - changing diffusion constants, do -not-diffuse to types etc...
	// Coupling coefficients cannot be changed and also there is no way to allocate extra fields while simulation is running

	if(potts->getDisplayUnitsFlag()){
		Unit diffConstUnit=powerUnit(potts->getLengthUnit(),2)/potts->getTimeUnit();
		Unit decayConstUnit=1/potts->getTimeUnit();
		Unit secretionConstUnit=1/potts->getTimeUnit();

		if (_xmlData->getFirstElement("AutoscaleDiffusion")){
			autoscaleDiffusion=true;
		}

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

	solverSpecific(_xmlData);
	diffSecrFieldTuppleVec.clear();
	bcSpecVec.clear();
	bcSpecFlagVec.clear();

	CC3DXMLElementList diffFieldXMLVec=_xmlData->getElements("DiffusionField");

	
	for(unsigned int i = 0 ; i < diffFieldXMLVec.size() ; ++i ){
		diffSecrFieldTuppleVec.push_back(DiffusionSecretionDiffusionFEFieldTupple<Cruncher>());
		DiffusionData & diffData=diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size()-1].diffData;
		SecretionData & secrData=diffSecrFieldTuppleVec[diffSecrFieldTuppleVec.size()-1].secrData;

		if(diffFieldXMLVec[i]->findElement("DiffusionData"))
			diffData.update(diffFieldXMLVec[i]->getFirstElement("DiffusionData"));

		if(diffFieldXMLVec[i]->findElement("SecretionData"))
			secrData.update(diffFieldXMLVec[i]->getFirstElement("SecretionData"));

		if(diffFieldXMLVec[i]->findElement("ReadFromFile"))
			readFromFileFlag=true;

		//boundary conditions parsing
		bcSpecFlagVec.push_back(false);
		bcSpecVec.push_back(BoundaryConditionSpecifier());

		if (diffFieldXMLVec[i]->findElement("BoundaryConditions")){
			bcSpecFlagVec[bcSpecFlagVec.size()-1]=true;
			BoundaryConditionSpecifier & bcSpec = bcSpecVec[bcSpecVec.size()-1];

			CC3DXMLElement * bcSpecElem = diffFieldXMLVec[i]->getFirstElement("BoundaryConditions");
			CC3DXMLElementList planeVec = bcSpecElem->getElements("Plane");



			for(unsigned int ip = 0 ; ip < planeVec.size() ; ++ip ){
				ASSERT_OR_THROW ("Boundary Condition specification Plane element is missing Axis attribute",planeVec[ip]->findAttribute("Axis"));
				string axisName=planeVec[ip]->getAttribute("Axis");
				int index=0;
				if (axisName=="x" ||axisName=="X" ){
					index=0;
				}
				if (axisName=="y" ||axisName=="Y" ){
					index=2;
				}
				if (axisName=="z" ||axisName=="Z" ){
					index=4;
				}

				if (planeVec[ip]->findElement("Periodic")){
					bcSpec.planePositions[index]=BoundaryConditionSpecifier::PERIODIC;
					bcSpec.planePositions[index+1]=BoundaryConditionSpecifier::PERIODIC;
				}else {
					//if (planeVec[ip]->findElement("ConstantValue")){
					CC3DXMLElementList cvVec=planeVec[ip]->getElements("ConstantValue");
					CC3DXMLElementList cdVec=planeVec[ip]->getElements("ConstantDerivative");

					for (unsigned int v = 0 ; v < cvVec.size() ; ++v ){
						string planePos=cvVec[v]->getAttribute("PlanePosition");
						double value=cvVec[v]->getAttributeAsDouble("Value");
						changeToLower(planePos);
						if (planePos=="min"){
							bcSpec.planePositions[index]=BoundaryConditionSpecifier::CONSTANT_VALUE;
							bcSpec.values[index]=value;

						}else if (planePos=="max"){
							bcSpec.planePositions[index+1]=BoundaryConditionSpecifier::CONSTANT_VALUE;
							bcSpec.values[index+1]=value;
						}else{
							ASSERT_OR_THROW("PlanePosition attribute has to be either max on min",false);
						}

					}
					if (cvVec.size()<=1){
						for (unsigned int d = 0 ; d < cdVec.size() ; ++d ){
							string planePos=cdVec[d]->getAttribute("PlanePosition");
							double value=cdVec[d]->getAttributeAsDouble("Value");
							changeToLower(planePos);
							if (planePos=="min"){
								bcSpec.planePositions[index]=BoundaryConditionSpecifier::CONSTANT_DERIVATIVE;
								bcSpec.values[index]=value;

							}else if (planePos=="max"){
								bcSpec.planePositions[index+1]=BoundaryConditionSpecifier::CONSTANT_DERIVATIVE;
								bcSpec.values[index+1]=value;
							}else{
								ASSERT_OR_THROW("PlanePosition attribute has to be either max on min",false);
							}

						}
					}

				}

			}

		}
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

	//variableDiffusionConstantFlagVec.assign(diffSecrFieldTuppleVec.size(),false);

	for(size_t i = 0 ; i < diffSecrFieldTuppleVec.size() ; ++i){
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
				diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&DiffusionSolverFE::secreteSingleField;
				++j;
			}
			else if((*sitr)=="SecretionOnContact"){
				diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&DiffusionSolverFE::secreteOnContactSingleField;
				++j;
			}
			else if((*sitr)=="ConstantConcentration"){
				diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j]=&DiffusionSolverFE::secreteConstantConcentrationSingleField;
				++j;
			}
		}
	}
}

template <class Cruncher>
std::string DiffusionSolverFE<Cruncher>::toString(){ //TODO: overload in cruncher?
	return "DiffusionSolverFE";
}

template <class Cruncher>
std::string DiffusionSolverFE<Cruncher>::steerableName(){
	return toString();
}


//The explicit instantiation part.
//Add new solvers here
template class DiffusionSolverFE<DiffusionSolverFE_CPU>; 
template class DiffusionSolverFE<DiffusionSolverFE_CPU_Implicit>; 

#if OPENCL_ENABLED == 1
template class DiffusionSolverFE<DiffusionSolverFE_OpenCL>;
//template class DiffusionSolverFE<DiffusionSolverFE_OpenCL_Implicit>;
template class DiffusionSolverFE<ReactionDiffusionSolverFE_OpenCL_Implicit>;
#endif
