#include "NonlinearSolver.h"

#include "ImplicitMatrix.h"
#include <sstream>
#include <CompuCell3D/CC3DExceptions.h>
#include "OpenCLKernel.h"
#include <cassert>
#include <numeric>
#include <limits>
#include "../GPUSolverParams.h"
#include <viennacl/linalg/norm_2.hpp>
#include <Logger/CC3DLogger.h>

//Theoretically, we need to use Biconjugate Stabilized Gradients method here,
//as the matrix is not symmetric.
//In practice, it's the best method to use indeed (checked for FitzHughNagumo model).
//Plain CG doesn't crash, but converges slower
//GMRES crashes, trying to converge

#include <viennacl/linalg/bicgstab.hpp>

#include <sstream>

#include "Helper.h"
#include "../MyTime.h"

using namespace CompuCell3D;

NonlinearSolver::NonlinearSolver(OpenCLHelper const &oclHelper, /*ImplicitMatrix &im,*/ std::vector<fieldNameAddTerm_t> const &fnats,  
	std::vector<UniSolverParams> const &solverParams, cl_mem const &d_cellTypes, GPUBoundaryConditions const &boundaryConditions,
	unsigned char fieldsCount_, std::string const &pathToKernels):
Solver(oclHelper, solverParams, d_cellTypes, boundaryConditions, fieldsCount_, pathToKernels),
mv_newField(fieldLength(&solverParams[0])*fieldsCount_),
    m_linearIterationsMade(0)
{
	CC3D_Log(LOG_DEBUG) << "NonlinearSolver::ctor";
	assert(fieldsCount_==fnats.size());
	assert(solverParams.size()==fnats.size());

    m_tmpReactionKernelsFN = genReactionKernels(fnats);
    std::cout << "reaction kernels created at file " << m_tmpReactionKernelsFN << std::endl;

	//loading OpenCL program
	std::string commonFN=pathToKernels+"common.cl";
	const char *programPaths[]={commonFN.c_str(), m_tmpReactionKernelsFN.c_str()};//TODO: find size of an array automatically
	CC3D_Log(LOG_DEBUG) << "OpenCL kernel names for NonlinearSolver:";
    for (int i = 0; i < 2; ++i) {
        CC3D_Log(LOG_DEBUG) << "\t"<<programPaths[i];
    }

    if (!oclHelper.LoadProgram(programPaths, 2, m_clProgram)) {
        throw std::runtime_error("Can't create NonlinearSolver object, OpenCL program creation failed");
    }

    int fl = fieldLength(&solverParams[0]);
    for (unsigned char i = 0; i < getFieldsCount(); ++i) {
        cl_mem outBuffer = getOutBuffer();

        cl_buffer_region subBufferInfo = {i * fl * sizeof(float), fl * sizeof(float)};
        cl_int retCode;

        cl_mem subBuffer = clCreateSubBuffer(mv_newField.handle().opencl_handle().get(), CL_MEM_READ_WRITE,
                                             CL_BUFFER_CREATE_TYPE_REGION, &subBufferInfo, &retCode);
        ASSERT_OR_THROW("Can't make a new subbuffer", retCode == CL_SUCCESS);
        m_newFieldSubBuffers.push_back(subBuffer);

    }

    m_minusG = new OpenCLKernel(m_clProgram, "minusG");
    m_Jacobian = new OpenCLKernel(m_clProgram, "Jacobian");
}


NonlinearSolver::~NonlinearSolver(void) {
    clReleaseProgram(m_clProgram);
    delete m_minusG;
    delete m_Jacobian;
    //remove(m_tmpReactionKernelsFN.c_str());
}

/*
void NonlinearSolver::for_each_im(std::function<void(ImplicitMatrix *)> const &fn){
	for(unsigned char i=0; i<m_fieldsCount; ++i)
		fn(m_ims[i].get());
}*/

bool myIsFinite(double f) {
#ifdef _WIN32
    return (_finite(f)!=false);
#else
    return std::isfinite(f);
#endif
}

float NonlinearSolver::Epsilon(viennacl::vector<float> const &v, viennacl::vector<float> const &newField) const {

    //AnalyzeVector(dim, "Krylov vector: ", v);
    //Real_t b=sqrt(std::numeric_limits<Real_t>::epsilon());
    float eMach = std::numeric_limits<float>::epsilon();

    float vn2 = viennacl::linalg::norm_2(v);

    //AnalyzeVector("v in Epsilon: ", v);

    if (!myIsFinite(vn2)) {
        analyze("v in Epsilon: ", v, getDim(), getFieldsCount());
        std::cout << std::endl;
        std::stringstream sstr;
        sstr << "Singular Krylov vector passed at linear iteration #" << m_linearIterationsMade
             << ". Try to reduce time step.";
        throw NewtonException(sstr.str().c_str());
    }

    if (vn2 == 0)
        return sqrt(eMach);
    //AnalyzeVector("NonLinear::Epsilon, newField",newField);
    float un2 = viennacl::linalg::norm_2(newField);//TODO: cache this!!

    assert(myIsFinite(un2));
    assert(un2 >= 0);

    float res = sqrt((1 + un2) * eMach) / vn2;
    //cout<<"sqrt arg: "<<(1+un2)*eMach;
    assert(myIsFinite(res));
    assert(res > 0);
    return res;

}


//compute as oldField+dt*NonLinear(newField)-prod(newField)
viennacl::vector<float> const &NonlinearSolver::minusGoalFunc(viennacl::vector<float> const &v_oldField) const {

    for (unsigned char i = 0; i < getFieldsCount(); ++i) {
        prodField(i, m_newFieldSubBuffers[i]);
    }


    Dim3D dim = getDim();

    cl_int3 clDim = {dim.x, dim.y, dim.z};

    cl_int stride = getFieldLength();

    try {
        m_minusG->setArgument(0, getTimeStep());
        m_minusG->setArgument(1, v_oldField.handle().opencl_handle().get());
        m_minusG->setArgument(2, mv_newField.handle().opencl_handle().get());
        m_minusG->setArgument(3, clDim);
        m_minusG->setArgument(4, getOutBuffer());
        m_minusG->setArgument(5, stride);
    } catch (std::exception &ec) {
        ASSERT_OR_THROW(ec.what(), false);
    }

    //size_t glob_size[]={f};
    size_t glob_size[] = {(size_t) dim.x, (size_t) dim.y, (size_t) dim.z};
    cl_int err = getOCLHelper().EnqueueNDRangeKernel(m_minusG->getKernel(), 3, glob_size, NULL);
    if (err != CL_SUCCESS) {
        std::stringstream sstr;
        sstr << "Can't compute goal function: " << getOCLHelper().ErrorString(err);
        ASSERT_OR_THROW(sstr.str().c_str(), false);
    }

    return getOutVector();
}


//compute as prod(updateVector)-dt*(Nonlinear(newField+e*updateVector)-Nonlinear(newField))/e
viennacl::vector<float> const &NonlinearSolver::prod(viennacl::vector<float> const &v_update) const {
    //m_ims[0]->prod(v_update.handle().opencl_handle());
    //for_each_im<cl_mem>(std::bind2nd(std::mem_fn(&ImplicitMatrix::prod), v_update.handle().opencl_handle()));
    //size_t totalLength=m_ims[0]->fieldLength();
    size_t totalLength = getFieldLength();
    for (unsigned char i = 0; i < getFieldsCount(); ++i) {
        cl_buffer_region subBufferInfo = {i * totalLength * sizeof(float), totalLength * sizeof(float)};
        cl_int retCode;
        cl_mem subBuffer = clCreateSubBuffer(v_update.handle().opencl_handle().get(), CL_MEM_READ_WRITE,
                                             CL_BUFFER_CREATE_TYPE_REGION, &subBufferInfo, &retCode);
        ASSERT_OR_THROW("Can't make an update subbuffer", retCode == CL_SUCCESS);
        prodField(i, subBuffer);
        clReleaseMemObject(subBuffer);
    }
    //return v_out;
    //float epsilon=1e-3;

    float epsilon = Epsilon(v_update, mv_newField);
    //std::cout<<"epsilon: "<<epsilon<<std::endl;
    Dim3D dim = getDim();

    cl_int3 clDim = {dim.x, dim.y, dim.z};

    cl_int stride = getFieldLength();
    try {
        m_Jacobian->setArgument(0, getTimeStep());
        m_Jacobian->setArgument(1, epsilon);
        m_Jacobian->setArgument(2, mv_newField.handle().opencl_handle().get());
        m_Jacobian->setArgument(3, v_update.handle().opencl_handle().get());
        m_Jacobian->setArgument(4, clDim);
        m_Jacobian->setArgument(5, getOutBuffer());
        m_Jacobian->setArgument(6, stride);

    } catch (std::exception &ec) {
        ASSERT_OR_THROW(ec.what(), false);
    }

    //size_t glob_size[]={f};
    size_t glob_size[] = {(size_t) dim.x, (size_t) dim.y, (size_t) dim.z};
    cl_int err = getOCLHelper().EnqueueNDRangeKernel(m_Jacobian->getKernel(), 3, glob_size, NULL);
    if (err != CL_SUCCESS) {
        std::stringstream sstr;
        sstr << "Can't compute jacobian function: " << getOCLHelper().ErrorString(err);
        ASSERT_OR_THROW(sstr.str().c_str(), false);
    }

    ++m_linearIterationsMade;

    return getOutVector();
}


viennacl::vector<float> const &
NonlinearSolver::NewField(float dt, viennacl::vector<float> const &v_oldField, NLSParams const &nlsParams) {

    assert(v_oldField.size() == mv_newField.size());

    mv_newField.clear();
    setTimeStep(dt);

    //cl_mem oldField=v_oldField.handle().opencl_handle();
    //cl_mem newField=mv_newField.handle().opencl_handle();
    Dim3D dim3d = getDim();

    float xNorm2 = viennacl::linalg::norm_2(v_oldField);

    float prevMguNorm2 = std::numeric_limits<float>::max();
    float prevShNorm2 = std::numeric_limits<float>::max();

    int shNorm2GrowCount = 0;
    int goalFNGrowCount = 0;

    //float mguNorm2=std::numeric_limits<float>::max();
    for (size_t i = 0; i < nlsParams.newton_.maxIterations_; ++i) {
        /*analyze("old field:\n", v_oldField, dim3d, m_fieldsCount);
		std::cout<<"\n";
		analyze("new field:\n", mv_newField, dim3d, m_fieldsCount);
		std::cout<<"\n";*/
        viennacl::vector<float> const &mGU = minusGoalFunc(v_oldField);
        //analyze("minus gu:\n", mGU, dim3d, m_fieldsCount);

        float mguNorm2 = viennacl::linalg::norm_2(mGU);
        //std::cout<<"; second norm: "<<mguNorm2<<std::endl;

        if (mguNorm2 <= nlsParams.newton_.fTol_) {
            //std::cout<<"requested tolerance achieved (1), norm_2(G)="<<viennacl::linalg::norm_2(minusGoalFunc(v_oldField))<<"\n";
            //std::cout<<"requested tolerance achieved (1)\n";
            return mv_newField;
        }

        if (mguNorm2 >= prevMguNorm2) {
            if (goalFNGrowCount == 0)
                CC3D_Log(LOG_DEBUG) << "Warning: second norm for goal function grows.";
			if(nlsParams.newton_.stopIfFTolGrows_){
				CC3D_Log(LOG_DEBUG) <<" Exit requested.";
				return mv_newField;
			}else{
				if(goalFNGrowCount==0)
					CC3D_Log(LOG_DEBUG) << " Continue solving.";
			}
			++goalFNGrowCount;
		}
		prevMguNorm2=mguNorm2;

        applyBCToRHS(mGU);

        MyTime::Time_t stepBT = MyTime::CTime();
        m_linearIterationsMade = 0;
        viennacl::linalg::bicgstab_tag solver_tag = viennacl::linalg::bicgstab_tag(nlsParams.linear_.tol_,
                                                                                   nlsParams.linear_.maxIterations_);
        viennacl::vector<float> const &outputField = viennacl::linalg::solve(*this,
                                                                             mGU,
                                                                             solver_tag);

        addSolvingTime(MyTime::ElapsedTime(stepBT, MyTime::CTime()));
        //std::cout<<"Solved in "<<solver_tag.iters()<<" iterations; achieved tolerance: "<<solver_tag.error()<<std::endl;
        //analyze("Update field:\n", outputField, dim3d, getFieldsCount());

        float shNorm2 = viennacl::linalg::norm_2(outputField);
        //std::cout<<"; second norm: "<<shNorm2<<std::endl;

        if (solver_tag.max_iterations() == solver_tag.iters()) {
            CC3D_Log(LOG_DEBUG) << "Warning: Linear solver didn't converge in "<<solver_tag.max_iterations()<<" iterations.";
			if(nlsParams.linear_.stopIfDidNotConverge_){
				CC3D_Log(LOG_DEBUG) << " Exit requested.";
				return mv_newField;
			}else{
				CC3D_Log(LOG_DEBUG) << " Continue solving.";
			}
		}

        if (shNorm2 <= nlsParams.newton_.stpTol_ * xNorm2) {
            //std::cout<<"requested tolerance achieved (2), norm_2(G)="<<viennacl::linalg::norm_2(minusGoalFunc(v_oldField))<<"\n";
            //std::cout<<"requested tolerance achieved (2)\n";

            return mv_newField;
        }

        if (shNorm2 >= prevShNorm2) {
            if (shNorm2GrowCount == 0)
                CC3D_Log(LOG_DEBUG) << "Warning: second norm for update vector grows.";
			if(nlsParams.newton_.stopIfFTolGrows_){
				CC3D_Log(LOG_DEBUG) << " Exit requested.";
				return mv_newField;
			}else{
				if(shNorm2GrowCount==0)
					CC3D_Log(LOG_DEBUG) << " Continue solving.";
			}
			++shNorm2GrowCount;
		}
		
		mv_newField+=outputField;
		//analyze("New field:\n", mv_newField, dim3d, getFieldsCount());
		//std::cout<<std::endl<<std::endl;
	}

    ASSERT_OR_THROW("Can't solve nonlinear system, all iterations used", true);
    return mv_newField;
}
