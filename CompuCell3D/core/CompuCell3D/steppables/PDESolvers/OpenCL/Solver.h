#ifndef SOLVER_H
#define SOLVER_H

#include <viennacl/vector.hpp>

#include "CompuCell3D/Field3D/Dim3D.h"

struct UniSolverParams;
struct GPUBoundaryConditions;

namespace CompuCell3D {

    class ImplicitMatrix;

    class OpenCLHelper;

    struct NLSParams {
        struct Linear {
            size_t maxIterations_;
            float tol_;
            bool stopIfDidNotConverge_;

            explicit Linear(size_t maxIterations = 400, float tol = 1e-8f, bool stopIfDidNotConverge = false) :
                    maxIterations_(maxIterations), tol_(tol), stopIfDidNotConverge_(stopIfDidNotConverge) {}
        } linear_;

        struct Newton {
            size_t maxIterations_;
            float fTol_;
            float stpTol_;
            bool stopIfFTolGrows_;

            explicit Newton(size_t maxIterations, float fTol, float stpTol, bool stopIfFTolGrows) :
                    maxIterations_(maxIterations), fTol_(fTol), stpTol_(stpTol), stopIfFTolGrows_(stopIfFTolGrows) {}
        } newton_;

        NLSParams(Linear linear, Newton newton) : linear_(linear), newton_(newton) {}
    };

//base class for both linear and nonlinear solvers
    class Solver {

        const unsigned char m_fieldsCount;
        std::vector<ImplicitMatrix const *> m_ims;
        viennacl::vector<float> mv_outputField;
        OpenCLHelper const &m_oclHelper;
        float m_dt;

        mutable float m_linearST;

        std::vector <cl_mem> m_outFieldSubBuffers;//need to store them to release later.

    protected:
        void setTimeStep(float dt);

        float getTimeStep() const { return m_dt; }

        Dim3D getDim() const;

        unsigned char getFieldsCount() const { return m_fieldsCount; }

        cl_mem getOutBuffer() const { return mv_outputField.handle().opencl_handle().get(); }

        viennacl::vector<float> &getOutVector() { return mv_outputField; }

        viennacl::vector<float> const &getOutVector() const { return mv_outputField; }

        cl_mem prodField(int i, cl_mem v) const;

        void applyBCToRHSField(int i, cl_mem v) const;

        int getFieldLength() const;

        OpenCLHelper const &getOCLHelper() const;

        void addSolvingTime(float st) const { m_linearST += st; }//kind of a stupid const limitation...

    public:

        //it actually modifies rhs
        void applyBCToRHS(viennacl::vector<float> const &rhs) const;

        Solver(OpenCLHelper const &oclHelper,
               std::vector <UniSolverParams> const &solverParams, cl_mem const &d_cellTypes,
               GPUBoundaryConditions const &boundaryConditions,
               unsigned char fieldsCount, std::string const &pathToKernels);

        ~Solver();

        virtual viennacl::vector<float> const &
        NewField(float dt, viennacl::vector<float> const &oldField, NLSParams const &nlsParams) = 0;

        //to use in conjunction with ViennaCL
        virtual viennacl::vector<float> const &prod(viennacl::vector<float> const &v_update) const = 0;

        //time spent in linear solver, ms
        float getLinearSolvingTime() const { return m_linearST; }
    };

}

#endif
