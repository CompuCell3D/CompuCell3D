#pragma once

#if defined (__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else

#include <CL/opencl.h>

#endif

#include <stdexcept>
#include <viennacl/vector.hpp>
#include "OpenCLBuffer.h"
#include "ReactionKernels.h"
#include <functional>

#include "Solver.h"

struct UniSolverParams;
struct GPUBoundaryConditions;


namespace CompuCell3D {


    class OpenCLKernel;

    class NewtonException : public std::runtime_error {
    public:
        NewtonException(const char *msg) : std::runtime_error(msg) {}
    };

    class NonlinearSolver : public Solver {
        cl_program m_clProgram;

        std::string m_tmpReactionKernelsFN;

        OpenCLKernel const *m_minusG, *m_Jacobian;

        viennacl::vector<float> mv_newField;

        std::vector <cl_mem> m_newFieldSubBuffers;

        mutable unsigned int m_linearIterationsMade;

    private:
        viennacl::vector<float> const &minusGoalFunc(viennacl::vector<float> const &v_oldField) const;

        //viennacl::vector<float> const &jacobianProd(viennacl::vector<float> const& v_update)const;
        float Epsilon(viennacl::vector<float> const &v, viennacl::vector<float> const &newField) const;

        /*template <typename Res_t>
        void for_each_im(std::function<Res_t(ImplicitMatrix const*) > fn)const{
            for(unsigned char i=0; i<m_fieldsCount; ++i)
                fn(m_ims[i].get());
        }*/

    public:
        //overriding
        virtual viennacl::vector<float> const &prod(viennacl::vector<float> const &v_update) const;

        NonlinearSolver(OpenCLHelper const &oclHelper, /*ImplicitMatrix &im,*/
                        std::vector <fieldNameAddTerm_t> const &fnats,
                        std::vector <UniSolverParams> const &solverParams, cl_mem const &d_cellTypes,
                        GPUBoundaryConditions const &boundaryConditions,
                        unsigned char fieldsCount, std::string const &pathToKernels);

        viennacl::vector<float> const &
        NewField(float dt, viennacl::vector<float> const &oldField, NLSParams const &nlsParams);

        ~NonlinearSolver(void);
    };

}//namespace CompuCell3D

//telling ViennaCL how to compute prod(NonLinearSolver, vector)
//return a vector made based on internal ImplicitMatrix array
namespace viennacl {
    namespace linalg {
        inline
        viennacl::vector<float> prod(CompuCell3D::NonlinearSolver const &nls, viennacl::vector<float> const &v) {
            return nls.prod(v);
        }
    }
}
