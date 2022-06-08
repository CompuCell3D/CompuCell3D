#ifndef DIFFUSIONSOLVERFE_CPU_H
#define DIFFUSIONSOLVERFE_CPU_H

#include "DiffusionSolverFE.h"
#include <CompuCell3D/CC3DEvents.h>

namespace CompuCell3D {

    class PDESOLVERS_EXPORT DiffusionSolverFE_CPU :
            public DiffusionSolverFE<DiffusionSolverFE_CPU>, public DiffusableVectorCommon<float, Array3DContiguous> {

    public:
        typedef Array3DContiguous<float> ConcentrationField_t;//TODO: check if I can automate this type deduction
        DiffusionSolverFE_CPU(void);

        virtual ~DiffusionSolverFE_CPU(void);

        //TODO: check if can use a constant diffData here
        // // // void diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData /*const*/ &diffData);
        // // // virtual void boundaryConditionInitImpl(int idx);
        virtual void handleEventLocal(CC3DEvent &_event);

        // Interface between Python and FluctuationCompensator

        // Call to update compensator for this solver before next compensation
        // Call this after modifying field values outside of core routine
        virtual void
        updateFluctuationCompensator() { if (fluctuationCompensator) fluctuationCompensator->updateTotalConcentrations(); }

    protected:
        //virtual void diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData &diffData);

        virtual void secreteSingleField(unsigned int idx);

        virtual void secreteOnContactSingleField(unsigned int idx);

        virtual void secreteConstantConcentrationSingleField(unsigned int idx);


        virtual void initImpl();

        virtual void extraInitImpl();

        virtual void initCellTypesAndBoundariesImpl();

        virtual void stepImpl(const unsigned int _currentStep);

        virtual void diffuseSingleField(unsigned int idx);

        virtual void solverSpecific(CC3DXMLElement *_xmlData);//reading solver-specific information from XML file
        virtual Dim3D getInternalDim();

        virtual void boundaryConditionInit(int idx);

        virtual std::string toStringImpl();

    private:
        void getMinMaxBox(bool useBoxWatcher, int threadNumber, Dim3D &minDim, Dim3D &maxDim) const;
        //void CheckConcentrationField(ConcentrationField_t &concentrationField)const;
    };

}//CompuCell3D 

#endif
