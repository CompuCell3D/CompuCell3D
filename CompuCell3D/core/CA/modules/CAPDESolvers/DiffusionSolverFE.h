#ifndef DIFFUSIONSOLVERFE_H
#define DIFFUSIONSOLVERFE_H


#include <string>
#include <CA/CASteppable.h>
#include "DiffusableVectorCommon.h"
#include "DiffSecrData.h"
#include "CAPDESolversDLLSpecifier.h"


namespace CompuCell3D {

class CAManager;

class CAPDESOLVERS_EXPORT DiffusionSolverFE :public CASteppable, 
	 public DiffusableVectorCommon<float, Array3DContiguous>
{
	
public:
	typedef Array3DContiguous<float> ConcentrationField_t;//TODO: check if I can automate this type deduction	
    DiffusionSolverFE(void);
	virtual ~DiffusionSolverFE(void);
    std::string printSolverName();
	void createFields(Dim3D _dim, std::vector<string> _fieldNamesVec);
	void init(CAManager *_caManager);

	void diffuseSingleField(int i=0);
	void secreteSingleField(int i=0);

	DiffusionData * getDiffusionData(std::string _fieldName);
	SecretionData * getSecretionData(std::string _fieldName);
	int findIndexForFieldName(std::string _fieldName);

	//CASteppable API
    virtual void start();
    virtual void step(const unsigned int currentStep) ;
	virtual std::string toString();
    
    

private:
	Dim3D workFieldDim;
	CAManager * caManager;
	std::vector<DiffusionData> diffDataVec;
	std::vector<SecretionData> secretionDataVec;
	std::map<std::string,unsigned int> fieldName2Index;
		
};

};//CompuCell3D 

#endif



