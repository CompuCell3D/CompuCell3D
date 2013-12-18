#ifndef DIFFUSIONSOLVERFE_CPU_H
#define DIFFUSIONSOLVERFE_CPU_H

#include "DiffusionSolverFE.h"
#include <CompuCell3D/CC3DEvents.h>

namespace CompuCell3D {

class PDESOLVERS_EXPORT DiffusionSolverFE_CPU :
	public DiffusionSolverFE<DiffusionSolverFE_CPU>, public DiffusableVectorCommon<float, Array3DContiguous>
{
	
public:
	typedef Array3DContiguous<float> ConcentrationField_t;//TODO: check if I can automate this type deduction
    DiffusionSolverFE_CPU(void);
	virtual ~DiffusionSolverFE_CPU(void);

	//TODO: check if can use a constant diffData here
	void diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData /*const*/ &diffData);
    virtual void handleEventLocal(CC3DEvent & _event);
		
protected:
	//virtual void diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData &diffData);
	virtual void initImpl();
	virtual void extraInitImpl();
	virtual void initCellTypesAndBoundariesImpl();
	virtual void solverSpecific(CC3DXMLElement *_xmlData);//reading solver-specific information from XML file
    virtual std::string toStringImpl();
private:
	//void CheckConcentrationField(ConcentrationField_t &concentrationField)const;
};

}//CompuCell3D 

#endif
