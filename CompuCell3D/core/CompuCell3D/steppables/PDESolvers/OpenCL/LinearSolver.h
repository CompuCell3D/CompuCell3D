#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include "Solver.h"

struct UniSolverParams;
struct GPUBoundaryConditions;
/*
namespace CompuCell3D {
class LinearSolver;
}


namespace viennacl{
	namespace linalg{
		vector<float> prod(CompuCell3D::LinearSolver const &ls, vector<float> const&v);
	}
}*/


namespace CompuCell3D {


    class LinearSolver : public Solver {

    public:
        LinearSolver(OpenCLHelper const &oclHelper,
                     std::vector <UniSolverParams> const &solverParams, cl_mem const &d_cellTypes,
                     GPUBoundaryConditions const &boundaryConditions,
                     unsigned char fieldsCount, std::string const &pathToKernels);

        //overriding
        virtual viennacl::vector<float> const &
        NewField(float, const viennacl::vector<float> &, const CompuCell3D::NLSParams &);

        //overriding
        virtual viennacl::vector<float> const &prod(viennacl::vector<float> const &v_update) const;

    };

}//namespace CompuCell3D

//telling ViennaCL how to compute prod(LinearSolver, vector)
//return a vector made based on internal ImplicitMatrix array
namespace viennacl {
    namespace linalg {
        inline
        viennacl::vector<float> prod(CompuCell3D::LinearSolver const &ls, viennacl::vector<float> const &v) {
            return ls.prod(v);
        }
    }
}

#endif
