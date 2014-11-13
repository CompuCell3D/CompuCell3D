#include "DiffusionSolverFE.h"
#include <CA/CAManager.h>
#include <CA/CACell.h>
#include <CA/CACellStack.h>


using namespace std;
using namespace CompuCell3D;


DiffusionSolverFE::DiffusionSolverFE(void):CASteppable(),DiffusableVectorCommon<float, Array3DContiguous>(),caManager(0),maxStableDiffConstant(0.23f),diffusionLatticeScalingFactor(1.0f),fieldIdxCounter(0)
{
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DiffusionSolverFE::~DiffusionSolverFE(void)
{
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::init(CAManager *_caManager){
	caManager = _caManager;
	
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::extraInit(){
	//called right before simulation run to finish initialization
    initializeSolver();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::initializeSolver(){

    
	unsigned int numberOfFields = diffDataVec.size();
	Dim3D dim = caManager->getCellFieldS()->getDim();

	this->allocateDiffusableFieldVector(numberOfFields ,dim); 
    

	//fieldName2Index.clear();

	for(unsigned int i=0 ; i < numberOfFields  ; ++i){
		//string fieldName = _fieldNamesVec[i];
        SecretionData & secrData = secretionDataVec[i];
        string fieldName = diffDataVec[i].name;
		this->setConcentrationFieldName(i,fieldName);
		caManager->registerConcentrationField(fieldName,getConcentrationField(i));

		//setting up fieldName2Index dictionary
		//fieldName2Index.insert(make_pair(fieldName,i));
		//this->getConcentrationField(i)->set(Point3D(10,10,i*10),20.0);
        for ( std::map<std::string, float>::iterator mitr = secrData.secretionMap.begin()  ; mitr != secrData.secretionMap.end() ; ++mitr){
            
            secrData.secretionConst[caManager->getTypeId(mitr->first)]=mitr->second;
        }

	}


	workFieldDim=this->getConcentrationField(0)->getInternalDim();

    //dealing with max diffusion constant
	maxStableDiffConstant=0.23;
	if(caManager->getBoundaryStrategy()->getLatticeType()==HEXAGONAL_LATTICE) {
		if (dim.x==1 || dim.y==1||dim.z==1){ //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D		
			maxStableDiffConstant=0.16f;
		}else{//3D
			maxStableDiffConstant=0.08f;
		}
	}else{//Square lattice
		if (dim.x==1 || dim.y==1||dim.z==1){ //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D				
			maxStableDiffConstant=0.23f;
		}else{//3D
			maxStableDiffConstant=0.14f;
		}
	}
	

	//determining latticeType and setting diffusionLatticeScalingFactor
	//When you evaluate div as a flux through the surface divided bby volume those scaling factors appear automatically. On cartesian lattife everythink is one so this is easy to forget that on different lattices they are not1
	diffusionLatticeScalingFactor=1.0;
	if (caManager->getBoundaryStrategy()->getLatticeType()==HEXAGONAL_LATTICE){
		if (dim.x==1 || dim.y==1||dim.z==1){ //2D simulation we ignore 1D simulations in CC3D they make no sense and we assume users will not attempt to run 1D simulations with CC3D
			diffusionLatticeScalingFactor=1.0f/sqrt(3.0f);// (2/3)/dL^2 dL=sqrt(2/sqrt(3)) so (2/3)/dL^2=1/sqrt(3)
		}else{//3D simulation
			diffusionLatticeScalingFactor=pow(2.0f,-4.0f/3.0f); //(1/2)/dL^2 dL dL^2=2**(1/3) so (1/2)/dL^2=1/(2.0*2^(1/3))=2^(-4/3)
		}

	}    


    //allocating  maxDiffConstVec and  scalingTimesPerMCSVec   
	scalingTimesPerMCSVec.assign(numberOfFields,0);
	maxDiffConstVec.assign(numberOfFields,0.0);
    
    //finding maximum diffusion coefficients for each field
    for(unsigned int i = 0 ; i < numberOfFields ; ++i){
		//for(int currentCellType = 0; currentCellType < UCHAR_MAX+1; currentCellType++) {
			//                 cout << "diffCoef[currentCellType]: " << diffSecrFieldTuppleVec[i].diffData.diffCoef[currentCellType] << endl;
		maxDiffConstVec[i] = (maxDiffConstVec[i] <  diffDataVec[i].diffConst) ? diffDataVec[i].diffConst: maxDiffConstVec[i];
    }
    
	Scale(maxDiffConstVec, maxStableDiffConstant);//TODO: remove for implicit solvers? 
    
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::Scale(std::vector<float> const &maxDiffConstVec, float maxStableDiffConstant){
   if (!maxDiffConstVec.size()){ //we will pass empty vector from update function . At the time of calling the update function we have no knowledge of maxDiffConstVec, maxStableDiffConstant
        return;
    }

	for(unsigned int i = 0; i < diffDataVec.size(); i++){
		scalingTimesPerMCSVec[i] = ceil(maxDiffConstVec[i]/maxStableDiffConstant); //compute number of calls to diffusion solver
		if (scalingTimesPerMCSVec[i]==0)
			continue;
        //diffusion data
        diffDataVec[i].timesPerMCS=scalingTimesPerMCSVec[i];
        diffDataVec[i].diffConst /= scalingTimesPerMCSVec[i];
        diffDataVec[i].decayConst /= scalingTimesPerMCSVec[i];		

        //secretion data
        std::map<std::string, float> & secretionMap = secretionDataVec[i].secretionMap;
        for (std::map<std::string, float>::iterator mitr = secretionMap.begin() ; mitr != secretionMap.end() ; ++mitr){
            mitr->second /= scalingTimesPerMCSVec[i];
        }

    }

    



}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::start(){
	cerr<<"INSIDE "<<toString()<<endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string DiffusionSolverFE::toString(){
	return string("DiffusionSolver");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string DiffusionSolverFE::printSolverName(){
    return string(" THIS IS DIFFUSION SOLVER FE");
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::printConfiguration(){

    for (int i  = 0 ; i < diffDataVec.size() ; ++i){
        cerr<<"FIELD NAME: "<<diffDataVec[i].name<<endl;
        cerr<<"diffConst "<<diffDataVec[i].diffConst<<endl;
        cerr<<"decayConst "<<diffDataVec[i].decayConst<<endl;
        cerr<<"************Secretion*************"<<endl;
        SecretionData & secrData = secretionDataVec[i];
        for (std::map<std::string, float>::iterator mitr =  secrData.secretionMap.begin() ; mitr !=  secrData.secretionMap.end(); ++ mitr){
            cerr<<"Secretion: type = "<<mitr->first<<" amount = "<<mitr->second<<endl;
        }
        cerr<<endl;
    }
        
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DiffusionSolverFE::addDiffusionAndSecretionData(std::string _fieldName){

    diffDataVec.push_back(DiffusionData());
    (--diffDataVec.end())->name = _fieldName;
    secretionDataVec.push_back(SecretionData());
    (--secretionDataVec.end())->name = _fieldName;
    fieldName2Index.insert(make_pair(_fieldName,fieldIdxCounter));
    bcSpecVec.push_back(CABoundaryConditionSpecifier());

    ++fieldIdxCounter;
    
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//SecretionData & DiffusionSolverFE::addSecretionData(){
//    secretionDataVec.push_back(SecretionData());
//    return *(--secretionDataVec.end());
//}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::addFields(std::vector<string> _fieldNamesVec){

	fieldNamesVec=_fieldNamesVec;

	unsigned int numberOfFields = _fieldNamesVec.size();
	Dim3D dim = caManager->getCellFieldS()->getDim();

	this->allocateDiffusableFieldVector(numberOfFields ,dim); 

	for (int i = 0 ;  i < _fieldNamesVec.size() ; ++i){
		cerr<<"GOT THIS FIELD NAME"<<_fieldNamesVec[i]<<endl;
	}

	//allocating default DiffusionData vector
	diffDataVec.clear();
	diffDataVec.assign(numberOfFields ,DiffusionData());

	//allocating default SecretionData vector
	secretionDataVec.clear();
	secretionDataVec.assign(numberOfFields ,  SecretionData() );

	fieldName2Index.clear();

	for(unsigned int i=0 ; i < numberOfFields  ; ++i){
		string fieldName = _fieldNamesVec[i];
		this->setConcentrationFieldName(i,fieldName);
		caManager->registerConcentrationField(fieldName,getConcentrationField(i));

		//setting up fieldName2Index dictionary
		fieldName2Index.insert(make_pair(fieldName,i));
		//this->getConcentrationField(i)->set(Point3D(10,10,i*10),20.0);
	}


	workFieldDim=this->getConcentrationField(0)->getInternalDim();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::createFields(Dim3D _dim, std::vector<string> _fieldNamesVec){
	cerr<<"Dim3D = "<<_dim<<endl;
	fieldNamesVec=_fieldNamesVec;
	for (int i = 0 ;  i < _fieldNamesVec.size() ; ++i){
		cerr<<"GOT THIS FIELD NAME"<<_fieldNamesVec[i]<<endl;
	}

	unsigned int numberOfFields = _fieldNamesVec.size();

	this->allocateDiffusableFieldVector(numberOfFields ,_dim); 

	//allocating default DiffusionData vector
	diffDataVec.clear();
	diffDataVec.assign(numberOfFields ,DiffusionData());

	//allocating default SecretionData vector
	secretionDataVec.clear();
	secretionDataVec.assign(numberOfFields ,  SecretionData() );

	fieldName2Index.clear();

	for(unsigned int i=0 ; i < numberOfFields  ; ++i){
		string fieldName = _fieldNamesVec[i];
		this->setConcentrationFieldName(i,fieldName);
		caManager->registerConcentrationField(fieldName,getConcentrationField(i));

		//setting up fieldName2Index dictionary
		fieldName2Index.insert(make_pair(fieldName,i));
		//this->getConcentrationField(i)->set(Point3D(10,10,i*10),20.0);
	}


	workFieldDim=this->getConcentrationField(0)->getInternalDim();

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int DiffusionSolverFE::findIndexForFieldName(std::string _fieldName){
	std::map<std::string,unsigned int>::iterator mitr = fieldName2Index.find(_fieldName);
	if(mitr != fieldName2Index.end()){
		return mitr->second;
	}
	return -1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DiffusionData * DiffusionSolverFE::getDiffusionData(std::string _fieldName){
	int idx = findIndexForFieldName(_fieldName);
	
	if(idx>=0){
		return & diffDataVec[idx];
	}

	return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

SecretionData * DiffusionSolverFE::getSecretionData(std::string _fieldName){
	int idx = findIndexForFieldName(_fieldName);
	
	if(idx>=0){
		return & secretionDataVec[idx];
	}

	return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CABoundaryConditionSpecifier * DiffusionSolverFE::getBoundaryConditionData(std::string _fieldName){
	int idx = findIndexForFieldName(_fieldName);
	
	if(idx>=0){
		return & bcSpecVec[idx];
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::diffuseSingleField(int i){

		DiffusionData & diffData=diffDataVec[i];
		float currentDiffCoef=diffData.diffConst;
		float currentDecayCoef=diffData.decayConst;
		//cerr<<"fieldName="<<getConcentrationFieldName(i)<<endl;
		//cerr<<"diffData.diffConst="<<diffData.diffConst<<endl;
		//cerr<<"diffData.decayConst="<<diffData.decayConst<<endl;
		//float currentDiffCoef=0.1;
		//float currentDecayCoef=0.0001;
		
		float currentConcentration = 0.0;
		float updatedConcentration,concentrationSum,varDiffSumTerm;
		//cerr<<"i="<<i<<endl;
		
		/*Array3DContiguous<float> & concentrationField =(Array3DContiguous<float>) *(this->getConcentrationField(i));*/
		Array3DContiguous<float>  * concentrationFieldPtr =(Array3DContiguous<float> *) this->getConcentrationField(i);
		Array3DContiguous<float> & concentrationField = *concentrationFieldPtr;
		
		Point3D pt;
		//cerr<<"workFieldDim="<<workFieldDim<<endl;
		//cerr<<"concentrationField.getDim()="<<concentrationField.getDim()<<endl;
        for (int z = 1; z < workFieldDim.z-1; z++)
            for (int y = 1; y < workFieldDim.y-1; y++)
                for (int x = 1; x < workFieldDim.x-1; x++){
					pt=Point3D(x-1,y-1,z-1);
					//cerr<<"pt="<<pt<<endl;
                    currentConcentration = concentrationField.getDirect(x,y,z);


                    

                    updatedConcentration=0.0;
                    concentrationSum=0.0;
                    varDiffSumTerm=0.0;

                    const std::vector<Point3D> & offsetVecRef=boundaryStrategy->getOffsetVec(pt);
                    for (register int i = 0  ; i<=maxNeighborIndex /*offsetVec.size()*/ ; ++i ){
                        const Point3D & offset = offsetVecRef[i];

                        concentrationSum += concentrationField.getDirect(x+offset.x,y+offset.y,z+offset.z);

                    }

                    concentrationSum -= (maxNeighborIndex+1)*currentConcentration;
                        
                    concentrationSum*=currentDiffCoef;

					updatedConcentration=(concentrationSum+varDiffSumTerm)+(1.0-currentDecayCoef)*currentConcentration;

					concentrationField.setDirectSwap(x,y,z,updatedConcentration);//updating scratch
				}

				concentrationField.swapArrays();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::secreteSingleField(int i){

		SecretionData & secrData=secretionDataVec[i];
		float currentConcentration = 0.0;
		float updatedConcentration,concentrationSum,varDiffSumTerm;

		
		/*Array3DContiguous<float> & concentrationField =(Array3DContiguous<float>) *(this->getConcentrationField(i));*/
		Array3DContiguous<float>  * concentrationFieldPtr =(Array3DContiguous<float> *) this->getConcentrationField(i);
		Array3DContiguous<float> & concentrationField = *concentrationFieldPtr;

		Field3D<CACellStack *> * cellField = caManager->getCellFieldS();
		CACellStack * cellStack;
		

        for (int z = 1; z < workFieldDim.z-1; z++)
            for (int y = 1; y < workFieldDim.y-1; y++)
                for (int x = 1; x < workFieldDim.x-1; x++){
					Point3D pt(x-1,y-1,z-1);
					cellStack = cellField->get(pt);
					
					if (!cellStack) continue;

					currentConcentration = concentrationField.getDirect(x,y,z);

					unsigned int numCells = cellStack->getNumCells();
					//cerr<<"fillLevel="<<fillLevel<<" pt="<<pt<<endl;
					//cerr<<"currentConcentration="<<currentConcentration<<endl;

					for (unsigned int i = 0 ; i < numCells ; ++i){
						CACell * cell = cellStack->getCellByIdx(i);
						//cerr<<"cell->type="<<(int)cell->type<<endl;
						//cerr<<"secr const="<<secrData.secretionConst[cell->type]<<endl;
						currentConcentration += secrData.secretionConst[cell->type];
					}

					concentrationField.setDirect(x,y,z,currentConcentration);
				}


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::boundaryConditionInit(int idx){
    // static_cast<Cruncher *>(this)->boundaryConditionInitImpl(idx);
    // return;
        
	ConcentrationField_t & _array = *this->getConcentrationField(idx);    
    
	Dim3D fieldDim = _array.getDim();
	CABoundaryConditionSpecifier & bcSpec=bcSpecVec[idx];
	//DiffusionData & diffData = diffSecrFieldTuppleVec[idx].diffData;
	//float deltaX=diffData.deltaX;
    float deltaX=1.0; // spacing is always 1.0 in CC3D solvers

	//ConcentrationField_t & _array=*concentrationField;
	
		//detailed specification of boundary conditions
		// X axis
		if (bcSpec.planePositions[0]==CABoundaryConditionSpecifier::PERIODIC || bcSpec.planePositions[1]==CABoundaryConditionSpecifier::PERIODIC){
			int x=0;
			for(int y=0 ; y< workFieldDim.y-1; ++y)
				for(int z=0 ; z<workFieldDim.z-1 ; ++z){
					_array.setDirect(x,y,z,_array.getDirect(fieldDim.x,y,z));
                    // cellTypeArray.setDirect(x,y,z,cellTypeArray.getDirect(fieldDim.x,y,z));
				}

				x=fieldDim.x+1;
				for(int y=0 ; y< workFieldDim.y-1; ++y)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,_array.getDirect(1,y,z));
                        // cellTypeArray.setDirect(x,y,z,cellTypeArray.getDirect(1,y,z));
					}

		}else{
            
			if (bcSpec.planePositions[0]==CABoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[0];
				int x=0;
				for(int y=0 ; y< workFieldDim.y-1; ++y)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[0]==CABoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
				float cdValue= bcSpec.values[0];
				int x=0;

				for(int y=0 ; y< workFieldDim.y-1; ++y)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,_array.getDirect(1,y,z)-cdValue*deltaX);
					}

			}

			if (bcSpec.planePositions[1]==CABoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[1];
				int x=fieldDim.x+1;
				for(int y=0 ; y< workFieldDim.y-1; ++y)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[1]==CABoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
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
		if (bcSpec.planePositions[2]==CABoundaryConditionSpecifier::PERIODIC || bcSpec.planePositions[3]==CABoundaryConditionSpecifier::PERIODIC){
			int y=0;
			for(int x=0 ; x< workFieldDim.x-1; ++x)
				for(int z=0 ; z<workFieldDim.z-1 ; ++z){
					_array.setDirect(x,y,z,_array.getDirect(x,fieldDim.y,z));
                    // cellTypeArray.setDirect(x,y,z,cellTypeArray.getDirect(x,fieldDim.y,z));
				}

				y=fieldDim.y+1;
				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,_array.getDirect(x,1,z));
                        // cellTypeArray.setDirect(x,y,z,cellTypeArray.getDirect(x,1,z));
					}

		}else{

			if (bcSpec.planePositions[2]==CABoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[2];
				int y=0;
				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[2]==CABoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
				float cdValue= bcSpec.values[2];
				int y=0;

				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,_array.getDirect(x,1,z)-cdValue*deltaX);
					}

			}

			if (bcSpec.planePositions[3]==CABoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[3];
				int y=fieldDim.y+1;
				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int z=0 ; z<workFieldDim.z-1 ; ++z){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[3]==CABoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
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
		if (bcSpec.planePositions[4]==CABoundaryConditionSpecifier::PERIODIC || bcSpec.planePositions[5]==CABoundaryConditionSpecifier::PERIODIC){
			int z=0;
			for(int x=0 ; x< workFieldDim.x-1; ++x)
				for(int y=0 ; y<workFieldDim.y-1 ; ++y){
					_array.setDirect(x,y,z,_array.getDirect(x,y,fieldDim.z));
                    // cellTypeArray.setDirect(x,y,z,cellTypeArray.getDirect(x,y,fieldDim.z));
				}

				z=fieldDim.z+1;
				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int y=0 ; y<workFieldDim.y-1 ; ++y){
						_array.setDirect(x,y,z,_array.getDirect(x,y,1));
                        // cellTypeArray.setDirect(x,y,z,cellTypeArray.getDirect(x,y,1));
					}

		}else{

			if (bcSpec.planePositions[4]==CABoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[4];
				int z=0;
				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int y=0 ; y<workFieldDim.y-1 ; ++y){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[4]==CABoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
				float cdValue= bcSpec.values[4];
				int z=0;

				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int y=0 ; y<workFieldDim.y-1 ; ++y){
						_array.setDirect(x,y,z,_array.getDirect(x,y,1)-cdValue*deltaX);
					}

			}

			if (bcSpec.planePositions[5]==CABoundaryConditionSpecifier::CONSTANT_VALUE){
				float cValue= bcSpec.values[5];
				int z=fieldDim.z+1;
				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int y=0 ; y<workFieldDim.y-1 ; ++y){
						_array.setDirect(x,y,z,cValue);
					}

			}else if(bcSpec.planePositions[5]==CABoundaryConditionSpecifier::CONSTANT_DERIVATIVE){
				float cdValue= bcSpec.values[5];
				int z=fieldDim.z+1;

				for(int x=0 ; x< workFieldDim.x-1; ++x)
					for(int y=0 ; y<workFieldDim.y-1 ; ++y){
						_array.setDirect(x,y,z,_array.getDirect(x,y,z-1)+cdValue*deltaX);
					}
			}

		}

	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::step(const unsigned int){

	int numberOfFields=diffDataVec.size();

	//cerr<<"This is step fcn numberOfFields="<<numberOfFields<<endl;
	
    for (int i  = 0 ; i < numberOfFields ; ++i){
            if (!scalingTimesPerMCSVec[i]){ //we do not call diffusion step but call secretion - this happens when diffusion const is 0 but we still want to have secretion
                secreteSingleField(i);
            }
            
            //cerr<<"scalingTimesPerMCSVec[i]="<<scalingTimesPerMCSVec[i]<<" diffDonst="<<this->diffDataVec[i].diffConst<<endl;
            for(int timesPerMCS = 0; timesPerMCS < scalingTimesPerMCSVec[i]; timesPerMCS ++) {
                
                boundaryConditionInit(i);//initializing boundary conditions
                diffuseSingleField(i);
                secreteSingleField(i);            
            }

    }


	//for (int i  = 0 ; i < numberOfFields ; ++i){
	//	secreteSingleField(i);
	//	diffuseSingleField(i);
	//}

}