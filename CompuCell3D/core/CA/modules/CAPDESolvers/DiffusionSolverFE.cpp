#include "DiffusionSolverFE.h"
#include <CA/CAManager.h>
#include <CA/CACell.h>
#include <CA/CACellStack.h>


using namespace std;
using namespace CompuCell3D;


DiffusionSolverFE::DiffusionSolverFE(void):DiffusableVectorCommon<float, Array3DContiguous>(),caManager(0)
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
void DiffusionSolverFE::createFields(Dim3D _dim, std::vector<string> _fieldNamesVec){
	cerr<<"Dim3D = "<<_dim<<endl;

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
		this->getConcentrationField(i)->set(Point3D(10,10,i*10),20.0);
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
void DiffusionSolverFE::diffuseSingleField(int i){

		DiffusionData & diffData=diffDataVec[i];
		float currentDiffCoef=diffData.diffConst;
		float currentDecayCoef=diffData.decayConst;
		cerr<<"fieldName="<<getConcentrationFieldName(i)<<endl;
		cerr<<"diffData.diffConst="<<diffData.diffConst<<endl;
		cerr<<"diffData.decayConst="<<diffData.decayConst<<endl;
		//float currentDiffCoef=0.1;
		//float currentDecayCoef=0.0001;
		
		float currentConcentration = 0.0;
		float updatedConcentration,concentrationSum,varDiffSumTerm;
		cerr<<"i="<<i<<endl;
		
		/*Array3DContiguous<float> & concentrationField =(Array3DContiguous<float>) *(this->getConcentrationField(i));*/
		Array3DContiguous<float>  * concentrationFieldPtr =(Array3DContiguous<float> *) this->getConcentrationField(i);
		Array3DContiguous<float> & concentrationField = *concentrationFieldPtr;
		
		Point3D pt;
		cerr<<"workFieldDim="<<workFieldDim<<endl;
		cerr<<"concentrationField.getDim()="<<concentrationField.getDim()<<endl;
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::step(int mcs){

	int numberOfFields=diffDataVec.size();

	cerr<<"This is step fcn numberOfFields="<<numberOfFields<<endl;
	

	for (int i  = 0 ; i < numberOfFields ; ++i){
		secreteSingleField(i);
		diffuseSingleField(i);
	}

}