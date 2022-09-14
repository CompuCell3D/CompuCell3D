#ifndef REACTION_DIFFUSION_SOLVER_FE_OPENCL_IMPLICIT
#define REACTION_DIFFUSION_SOLVER_FE_OPENCL_IMPLICIT

#include <CompuCell3D/CC3DEvents.h>
#include "../DiffusionSolverFE.h"

#include "DiffusableVectorRD.hpp"
#include "Solver.h"
#include "../GPUSolverParams.h"
#include "GPUBoundaryConditions.h"
#include "Solver.h"
#include "OpenCLBuffer.h"

namespace CompuCell3D {


    class ReactionDiffusionSolverFE_OpenCL_Implicit :
            public DiffusionSolverFE<ReactionDiffusionSolverFE_OpenCL_Implicit>,
            public DiffusableVectorRDOpenCLImpl<float> {
        static OpenCLHelper const *m_oclHelper;
        Solver *m_solver;
        viennacl::vector<float> *mv_inputField;
        OpenCLBuffer *md_cellTypes;//TODO: make this a member of NonlienarSolver
        NLSParams *m_nlsParams;

        float m_dt;//time step

        std::vector <UniSolverParams_t> mh_solverParams;

        GPUBoundaryConditions m_GPUbc;//TODO: must be an array of these!!!

        float m_solvingTime;


        size_t m_fieldLen;

        virtual std::string toStringImpl();

    private:
        //object factory to make solver of an appropriate class
        Solver *makeSolver() const;

    public:
        typedef DiffusableVectorRDOpenCLImplFieldProxy<float> ConcentrationField_t; //Dummy typedef
        ReactionDiffusionSolverFE_OpenCL_Implicit();

        ~ReactionDiffusionSolverFE_OpenCL_Implicit(void);

        void diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData const &diffData);

        virtual void handleEventLocal(CC3DEvent &_event);

        void finish();

    protected:
        virtual void stepImpl(
                const unsigned int _currentStep);//overriding it, as we solve all fields together due to the reaction part
        virtual void initImpl(void);

        virtual void extraInitImpl(void);

        virtual void initCellTypesAndBoundariesImpl(void);

        virtual void solverSpecific(CC3DXMLElement *);

        //if an array used for storing has an extra boundary layer around it
        virtual bool hasExtraLayer() const;

    };

}//namespace CompuCell3D

#endif

