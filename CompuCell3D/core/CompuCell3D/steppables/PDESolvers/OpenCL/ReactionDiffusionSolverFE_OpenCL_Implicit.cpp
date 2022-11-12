#include "ReactionDiffusionSolverFE_OpenCL_Implicit.h"
//#include <windows.h>//TODO: remove

#include "OpenCLHelper.h"
#include "OCLNeighbourIndsInfo.h"
#include "NonlinearSolver.h"
#include "LinearSolver.h"

#include "../MyTime.h"

#include <XMLUtils/CC3DXMLElement.h>

using namespace CompuCell3D;

OpenCLHelper const *ReactionDiffusionSolverFE_OpenCL_Implicit::m_oclHelper;

ReactionDiffusionSolverFE_OpenCL_Implicit::ReactionDiffusionSolverFE_OpenCL_Implicit(void) {
    CC3D_Log(LOG_DEBUG) << "Starting ReactionDiffusionSolverFE_OpenCL_Implicit ctor";

    if (!m_oclHelper) {
        m_oclHelper = new OpenCLHelper(0);//TODO: add gpu selector
        //attaching vieanncl context to the initialized one
        viennacl::ocl::setup_context(0, m_oclHelper->getContext(), m_oclHelper->getDevice(),
                                     m_oclHelper->getCommandQueue());
    }
}

void ReactionDiffusionSolverFE_OpenCL_Implicit::handleEventLocal(CC3DEvent &_event) {
    if (_event.id == LATTICE_RESIZE) {
        // CODE WHICH HANDLES CELL LATTICE RESIZE
    }
}

ReactionDiffusionSolverFE_OpenCL_Implicit::~ReactionDiffusionSolverFE_OpenCL_Implicit(void) {
    //delete m_oclHelper; m_oclHelper=NULL;
    delete m_solver;
    delete mv_inputField;
    delete md_cellTypes;
    delete m_nlsParams;
}


bool ReactionDiffusionSolverFE_OpenCL_Implicit::hasExtraLayer() const {
    return false;
}


void ReactionDiffusionSolverFE_OpenCL_Implicit::initImpl(void) {
    CC3D_Log(LOG_TRACE) << "ReactionDiffusionSolverFE_OpenCL_Implicit::initImpl, not implemented!!!";

    m_solvingTime = 0;
    for (int i = 0; i < getFieldsCount(); ++i) {
        BoundaryConditionSpecifier &bcSpec = bcSpecVec[i];
        if (periodicBoundaryCheckVector[0] == true) {
            bcSpec.planePositions[0] = BoundaryConditionSpecifier::PERIODIC;
            bcSpec.planePositions[1] = BoundaryConditionSpecifier::PERIODIC;
            bcSpec.values[0] = 0;
            bcSpec.values[1] = 0;
        }

        if (periodicBoundaryCheckVector[1] == true) {
            bcSpec.planePositions[2] = BoundaryConditionSpecifier::PERIODIC;
            bcSpec.planePositions[3] = BoundaryConditionSpecifier::PERIODIC;
            bcSpec.values[2] = 0;
            bcSpec.values[3] = 0;
        }

        if (periodicBoundaryCheckVector[2] == true) {
            bcSpec.planePositions[4] = BoundaryConditionSpecifier::PERIODIC;
            bcSpec.planePositions[5] = BoundaryConditionSpecifier::PERIODIC;
            bcSpec.values[4] = 0;
            bcSpec.values[5] = 0;
        }

    }

    //ASSERT_OR_THROW("not implemented", false);
}

void ReactionDiffusionSolverFE_OpenCL_Implicit::finish() {
    DiffusionSolverFE::finish();
    CC3D_Log(LOG_DEBUG) << m_solvingTime<<" ms spent for solving only";
    CC3D_Log(LOG_DEBUG) << m_solver->getLinearSolvingTime()<<" ms spent on solving linear systems";
}

std::ostream &operator<<(std::ostream &os, cl_int4 const &val) {
    os << "(";
    for (int i = 0; i < 3; ++i) {
        os << val.s[i] << " ";
    }
    os << val.s[3] << ")";
    return os;
}

Solver *ReactionDiffusionSolverFE_OpenCL_Implicit::makeSolver() const {

    if (hasAdditionalTerms()) {
        //additional terms
        std::vector <fieldNameAddTerm_t> fnats(fieldsCount());
        for (unsigned int i = 0; i < fieldsCount(); ++i) {
            DiffusionData const &diffData = diffSecrFieldTuppleVec[i].diffData;
            std::string name = getConcentrationFieldName(i);
            string addTerm = diffData.additionalTerm;
            if (addTerm.empty()) {
                addTerm = "return 0;";
            } else if (addTerm.find("return") == string::npos) {
                addTerm = "return " + addTerm + ";";
            }
            fnats[i] = make_pair(name, addTerm);
            CC3D_Log(LOG_DEBUG) << "Additional term: "<<fnats[i].first<<"/"<<fnats[i].second;
        }

        CC3D_Log(LOG_DEBUG) << "Making Nonlinear Solver";
        return new NonlinearSolver(*m_oclHelper, fnats, mh_solverParams, md_cellTypes->buffer(), m_GPUbc, fieldsCount(),
                                   "lib/CompuCell3DSteppables/OpenCL/");
    } else {

        CC3D_Log(LOG_DEBUG) << "No additional terms found; making Linear Solver";
        return new LinearSolver(*m_oclHelper, mh_solverParams, md_cellTypes->buffer(), m_GPUbc, fieldsCount(),
                                "lib/CompuCell3DSteppables/OpenCL/");
    }

}

void ReactionDiffusionSolverFE_OpenCL_Implicit::extraInitImpl(void) {
    CC3D_Log(LOG_TRACE) << "ReactionDiffusionSolverFE_OpenCL_Implicit::extraInitImpl, not implemented!!!";
	

    try {

        ASSERT_OR_THROW("For 2D case, the domain is allowed to be flat along the \"z\" axis only ",
                        fieldDim.x != 1 && fieldDim.y != 1);
        OCLNeighbourIndsInfo onii = OCLNeighbourIndsInfo::Init(latticeType, fieldDim, getBoundaryStrategy(),
                                                               hexOffsetArray, offsetVecCartesian);


        //preparing solver parameters
        mh_solverParams.resize(fieldsCount());
        //float other_dt;
        for (unsigned int i = 0; i < fieldsCount(); ++i) {

            DiffusionData &diffData = diffSecrFieldTuppleVec[i].diffData;

			/*if(i==0){
				other_dt=diffData.deltaT;
			}else{
				ASSERT_OR_THROW("All fields must have the same time step", other_dt==diffData.deltaT);
			}*/
			for( int j=0; j<UCHAR_MAX; ++j){
				mh_solverParams[i].diffCoef[j]=diffData.diffCoef[j];
				mh_solverParams[i].decayCoef[j]=diffData.decayCoef[j];
			}
			mh_solverParams[i].dx=deltaX;
			mh_solverParams[i].hexLattice=(latticeType==HEXAGONAL_LATTICE);
			mh_solverParams[i].nbhdConcLen=onii.m_nbhdConcLen;
			mh_solverParams[i].nbhdDiffLen=onii.m_nbhdDiffLen;
		
			mh_solverParams[i].xDim=fieldDim.x;
			mh_solverParams[i].yDim=fieldDim.y;
			mh_solverParams[i].zDim=fieldDim.z;
			CC3D_Log(LOG_DEBUG) << "Current size: "<<onii.mh_nbhdConcShifts.size();
            ASSERT_OR_THROW("Must be less or equal than 6 so far", onii.mh_nbhdConcShifts.size() <= 6);
            for (size_t j = 0; j < onii.mh_nbhdConcShifts.size(); ++j) {
                CC3D_Log(LOG_TRACE)<<"Current shift: "<<onii.mh_nbhdConcShifts[j];
				mh_solverParams[i].nbhdShifts[j]=onii.mh_nbhdConcShifts[j];
			}
		}

		m_dt=deltaT;
		CC3D_Log(LOG_DEBUG) << "Time step "<<m_dt<<" requested";

        Dim3D dim = getDim();

        m_fieldLen = dim.x * dim.y * dim.z;

        mv_inputField = new viennacl::vector<float>(m_fieldLen * fieldsCount());

        md_cellTypes = new OpenCLBuffer(*m_oclHelper, CL_MEM_READ_WRITE, m_fieldLen, NULL);


        for (int i = 0; i < 1; ++i)//TODO: add multiple bcs for every field
        {
            BoundaryConditionSpecifier &bcSpec = bcSpecVec[i];

			for(int j=0; j<6; ++j){
				m_GPUbc.planePositions[j]=static_cast<BCType>(bcSpec.planePositions[j]);
				m_GPUbc.values[j]=static_cast<float>(bcSpec.values[j]);
			}
		}
		CC3D_Log(LOG_DEBUG) << "start initializing";

        m_solver = makeSolver();
        CC3D_Log(LOG_DEBUG) << "extraInitImpl finished; m_nbhdConcLen="<<onii.m_nbhdConcLen<<"; m_nbhdDiffLen="<<onii.m_nbhdDiffLen;

    } catch (std::exception &ec) {
        ASSERT_OR_THROW(ec.what(), false);
    }
}


void ReactionDiffusionSolverFE_OpenCL_Implicit::stepImpl(const unsigned int _currentStep) {
    try {
        (this->*secretePtr)();

        initCellTypesAndBoundariesImpl();

        CC3D_Log(LOG_DEBUG) << "ReactionDiffusionSolverFE_OpenCL_Implicit::stepImpl: step #" << _currentStep;
        Dim3D dim = getDim();

        m_oclHelper->WriteBuffer(mv_inputField->handle().opencl_handle().get(), getPtr(), mv_inputField->size());

        MyTime::Time_t stepBT = MyTime::CTime();
        viennacl::vector<float> const &newField = m_solver->NewField(m_dt, *mv_inputField, *m_nlsParams);
        m_solvingTime += MyTime::ElapsedTime(stepBT, MyTime::CTime());

        m_oclHelper->ReadBuffer(newField.handle().opencl_handle().get(), getPtr(), mv_inputField->size());
    } catch (std::exception &ec) {
        ASSERT_OR_THROW(ec.what(), false);
    }
}


void ReactionDiffusionSolverFE_OpenCL_Implicit::initCellTypesAndBoundariesImpl(void) {
    cl_int err = m_oclHelper->WriteBuffer(md_cellTypes->buffer(), h_celltype_field->getContainer(), m_fieldLen);
    ASSERT_OR_THROW("Can not copy Cell Type field to GPU", err == CL_SUCCESS);
}

void ReactionDiffusionSolverFE_OpenCL_Implicit::solverSpecific(CC3DXMLElement *_xmlData) {
    CC3D_Log(LOG_DEBUG) << "ReactionDiffusionSolverFE_OpenCL_Implicit::solverSpecific";
    //ASSERT_OR_THROW("not implemented", false);

    if (_xmlData->findElement("DeltaX"))
        deltaX = static_cast<float>(_xmlData->getFirstElement("DeltaX")->getDouble());

    if (_xmlData->findElement("DeltaT")) {
        deltaT = static_cast<float>(_xmlData->getFirstElement("DeltaT")->getDouble());
        CC3D_Log(LOG_TRACE) << "************* another time step requested: "<<deltaT;
    }

    m_nlsParams = new NLSParams(NLSParams::Linear(), NLSParams::Newton(100, 1e-6f, 1e-6f, false));

    if (_xmlData->findElement("MaxLinearIterations")) {
        m_nlsParams->linear_.maxIterations_ = _xmlData->getFirstElement("MaxLinearIterations")->getInt();
    }

    if (_xmlData->findElement("LinearSolverTolerance")) {
        m_nlsParams->linear_.tol_ = static_cast<float>(_xmlData->getFirstElement("LinearSolverTolerance")->getDouble());
    }

    if (_xmlData->findElement("StopIfLinearSolverHasNotConverged")) {
        m_nlsParams->linear_.stopIfDidNotConverge_ = _xmlData->getFirstElement(
                "StopIfLinearSolverHasNotConverged")->getBool();
    }


    if (_xmlData->findElement("MaxNewtonIterations")) {
        m_nlsParams->newton_.maxIterations_ = _xmlData->getFirstElement("MaxNewtonIterations")->getInt();
    }

    if (_xmlData->findElement("NewtonFTolerance")) {
        m_nlsParams->newton_.fTol_ = static_cast<float>(_xmlData->getFirstElement("NewtonFTolerance")->getDouble());
    }

    if (_xmlData->findElement("NewtonSTPTolerance")) {
        m_nlsParams->newton_.stpTol_ = static_cast<float>(_xmlData->getFirstElement("NewtonSTPTolerance")->getDouble());
    }

    if (_xmlData->findElement("StopNewtonIfToleranceGrows")) {
        m_nlsParams->newton_.stopIfFTolGrows_ = _xmlData->getFirstElement("StopNewtonIfToleranceGrows")->getBool();
    }

}

//no need for an implementation 
void ReactionDiffusionSolverFE_OpenCL_Implicit::diffuseSingleFieldImpl(
        ConcentrationField_t &concentrationField, DiffusionData const &diffData) {
    //ASSERT_OR_THROW("not implemented", false);
}

std::string ReactionDiffusionSolverFE_OpenCL_Implicit::toStringImpl() {
    return "ReactionDiffusionSolverFE_OpenCL_Implicit";
}