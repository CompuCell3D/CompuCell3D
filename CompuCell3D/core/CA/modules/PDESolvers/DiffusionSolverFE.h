#ifndef DIFFUSIONSOLVERFE_H
#define DIFFUSIONSOLVERFE_H


#include <string>
#include "DiffusableVectorCommon.h"
#include "PDESolversDLLSpecifier.h"


namespace CompuCell3D {

class CAManager;

class PDESOLVERS_EXPORT DiffusionSolverFE :
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


private:
	Dim3D workFieldDim;
	CAManager * caManager;
		
};

};//CompuCell3D 

#endif



