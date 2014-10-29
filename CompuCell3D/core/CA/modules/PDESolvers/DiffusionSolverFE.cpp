#include "DiffusionSolverFE.h"
#include <CA/CAManager.h>
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

	this->allocateDiffusableFieldVector(_fieldNamesVec.size(),_dim); 

	for(unsigned int i=0 ; i < _fieldNamesVec.size() ; ++i){
		string fieldName = _fieldNamesVec[i];
		this->setConcentrationFieldName(i,fieldName);
		caManager->registerConcentrationField(fieldName,getConcentrationField(i));
		this->getConcentrationField(i)->set(Point3D(10,10,i*10),20.0);
	}


	workFieldDim=this->getConcentrationField(0)->getInternalDim();

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE::diffuseSingleField(int i){

		float currentDiffCoef=0.1;
		float currentDecayCoef=0.0001;
		
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