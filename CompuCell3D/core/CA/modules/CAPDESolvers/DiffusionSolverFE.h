#ifndef DIFFUSIONSOLVERFE_H
#define DIFFUSIONSOLVERFE_H


#include <string>
#include <CA/CASteppable.h>
#include "DiffusableVectorCommon.h"
#include "DiffSecrData.h"
#include "CABoundaryConditionSpecifier.h"
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
	void addFields(std::vector<string> _fieldNamesVec);

    void addDiffusionAndSecretionData(std::string _fieldName);

    void initializeSolver();

    void printConfiguration();
    //SecretionData & addSecretionData();

	void diffuseSingleField(int i=0);
	void secreteSingleField(int i=0);

	DiffusionData * getDiffusionData(std::string _fieldName);
	SecretionData * getSecretionData(std::string _fieldName);
    CABoundaryConditionSpecifier * getBoundaryConditionData(std::string _fieldName);

	int findIndexForFieldName(std::string _fieldName);
    void DiffusionSolverFE::boundaryConditionInit(int idx);

	//CASteppable API
	virtual void init(CAManager *_caManager);
	virtual void extraInit();
    virtual void start();
    virtual void step(const unsigned int currentStep) ;
	virtual std::string toString();
    
    

private:
    void Scale(std::vector<float> const &maxDiffConstVec, float maxStableDiffConstant);
	Dim3D workFieldDim;
	CAManager * caManager;
	std::vector<DiffusionData> diffDataVec;
	std::vector<SecretionData> secretionDataVec;
    std::vector<float> maxDiffConstVec;
    std::vector<float> scalingTimesPerMCSVec;

    std::vector<CABoundaryConditionSpecifier> bcSpecVec;

	std::vector<string> fieldNamesVec;
	std::map<std::string,unsigned int> fieldName2Index;
	float maxStableDiffConstant;
    float diffusionLatticeScalingFactor;
    int fieldIdxCounter;

};

};//CompuCell3D 

#endif



