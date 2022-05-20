#ifndef IMPLICIT_DIFF_MATRIX_H
#define IMPLICIT_DIFF_MATRIX_H

//class for handling implicit matrix-vector multiplication
//When we solve a sytem of linear equations Ax=b, there is typically no need
//to store matrix A explicitly in memory. The only operation that is required for many 
//linear system solvers is a matrix-vector multiplication and
//we can perform this multiplication on-the-fly based on known
//diffusion constants and boundary conditions.
//Here we decompose matrix-vector product into two sequential operatons,
//one for multiplying the "central part" of the domain and a separate
//kernel for adding boundary elements with respect to boundary conditions
//No dummy (ghost) elemetns outside domain are used.
//Ivan Komarov, 2012

#include "OpenCLHelper.h"
#include "OpenCLKernel.h"
#include "OpenCLBuffer.h"
#include "CompuCell3D/Field3D/Dim3D.h"

#include <memory>

struct UniSolverParams;

struct GPUBoundaryConditions;

namespace CompuCell3D {


    class ImplicitMatrix {
        UniSolverParams const &mh_solverParams;
        OpenCLHelper const &m_oclHelper;
        cl_program m_clProgram;
        OpenCLKernel const *m_prodCoreKernel, *m_prodBoundariesKernel, *m_modifyRHStoBC;
        cl_mem const &md_cellTypes;
        OpenCLBuffer md_solverParams;
        //OpenCLBuffer md_outputField; //where to store the results of A*x
        cl_mem md_outputField; //where to store the results of A*x
        OpenCLBuffer md_boundaryConditions;
        GPUBoundaryConditions const &mh_boundaryConditions;
        mutable bool isTimeStepSet;
        //float m_dt;

    private:
        //disable copying
        ImplicitMatrix(ImplicitMatrix const &);

        ImplicitMatrix &operator=(ImplicitMatrix const &);

    public:
        //initialize with OpenCLHelper, solver parameters and cell types
        ImplicitMatrix(OpenCLHelper const &oclHelper, UniSolverParams const &solverParams, cl_mem const &d_cellTypes,
                       cl_mem const &d_outputfield,
                       GPUBoundaryConditions const &boundaryConditions, std::string const &pathToKernels);

        void ApplyBCToRHS(cl_mem v) const;

        //need to change the time step independently
        void setdt(float dt) const;

        ~ImplicitMatrix();

        //length of a field in elements
        size_t fieldLength() const;

        //compute matrix-vector product and place it into md_outputField vector
        //returns md_outputField so that caller has an access to the output vector
        cl_mem prod(cl_mem v) const;

        Dim3D domainSize() const;

    private:

        //compute matrix-vector product for the internal part of the domain and place it into md_outputField vector
        void prodCore(cl_mem v) const;

        //compute matrix-vector product for the boundary layer and place it into md_outputField vector
        void prodBoundaries(cl_mem v) const;
    };

}


#endif