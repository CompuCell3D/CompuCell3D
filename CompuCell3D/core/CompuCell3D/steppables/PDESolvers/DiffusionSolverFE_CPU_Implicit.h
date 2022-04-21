#ifndef DIFFUSION_SOLVER_FE_CPU_IMPLICIT
#define DIFFUSION_SOLVER_FE_CPU_IMPLICIT

#include <CompuCell3D/CC3DEvents.h>

#include "DiffusionSolverFE.h"
#include <Eigen/Sparse>

#include <Eigen/Core>

namespace CompuCell3D {

    class DiffusionSolverFE_CPU_Implicit :
            public DiffusionSolverFE<DiffusionSolverFE_CPU_Implicit>,
            public DiffusableVectorCommon<float, Array3DContiguous> {
        typedef float Real_t;//to switch between float and double easily
        typedef Eigen::Matrix<Real_t, Eigen::Dynamic, 1> EigenRealVector;
        typedef Eigen::SparseMatrix <Real_t, Eigen::RowMajor> EigenSparseMatrix;
    public:
        typedef Array3DContiguous<float> ConcentrationField_t;//TODO: check if I can automate this type deduction
        DiffusionSolverFE_CPU_Implicit(void);

        ~DiffusionSolverFE_CPU_Implicit(void);

        //TODO: check if can use a constant diffData here
        void diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData /*const*/ &diffData);

        virtual void handleEventLocal(CC3DEvent &_event);

    protected:

        virtual void step(const unsigned int _currentStep);

        virtual void Scale(std::vector<float> const &maxDiffConstVec, float maxStableDiffConstant);

        virtual void initImpl();

        virtual void extraInitImpl();

        virtual void initCellTypesAndBoundariesImpl();

        virtual void solverSpecific(CC3DXMLElement *_xmlData);//reading solver-specific information from XML file
        virtual std::string toStringImpl();

    private:
        void Implicit(ConcentrationField_t const &concentrationField, DiffusionData const &diffData,
                      EigenRealVector const &b, EigenRealVector &x);
    };

}//namespace CompuCell3D

#endif

