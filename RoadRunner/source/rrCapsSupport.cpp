#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrBaseParameter.h"
#include "rrParameter.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrCapsSupport.h"
#include "rrCVODEInterface.h"
#include "rrCapability.h"

//---------------------------------------------------------------------------
namespace rr
{

CapsSupport::CapsSupport(RoadRunner* rr)
:
mName("RoadRunner"),
mDescription("Settings For RoadRunner"),
mRoadRunner(rr)
{
    if(mRoadRunner && mRoadRunner->getCVodeInterface())
    {
        CvodeInterface*  cvode = mRoadRunner->getCVodeInterface();
        Capability integration("integration", "CVODE", "CVODE Integrator");

        integration.add(new Parameter<int>(    "BDFOrder",     cvode->mMaxBDFOrder,     "Maximum order for BDF Method"));
        integration.add(new Parameter<int>(    "AdamsOrder",   cvode->mMaxAdamsOrder,   "Maximum order for Adams Method"));
        integration.add(new Parameter<double>( "rtol",         cvode->mRelTol,          "Relative Tolerance"));
        integration.add(new Parameter<double>( "atol",         cvode->mAbsTol,          "Absolute Tolerance"));
        integration.add(new Parameter<int>(    "maxsteps",     cvode->mMaxNumSteps,     "Maximum number of internal stepsc"));
        integration.add(new Parameter<double>( "initstep",     cvode->mInitStep,        "the initial step size"));
        integration.add(new Parameter<double>( "minstep",      cvode->mMinStep,         "specifies a lower bound on the magnitude of the step size."));
        integration.add(new Parameter<double>( "maxstep",      cvode->mMaxStep,         "specifies an upper bound on the	magnitude of the step size."));

        integration.add(new Parameter<bool>(   "conservation", mRoadRunner->computeAndAssignConservationLaws(),
        																					"enables (=true) or disables \
                                                                                            (=false) the conservation analysis \
                                                                                            of models for timecourse simulations."));
        //Add section to Capablities
        Add(integration);
    }

    if(mRoadRunner && mRoadRunner->getNLEQInterface())
    {
        NLEQInterface* solver = mRoadRunner->getNLEQInterface();
        Capability steady("SteadyState", "NLEQ2", "NLEQ2 Steady State Solver");
        steady.add(new Parameter<int>("MaxIterations", 			solver->maxIterations, "Maximum number of newton iterations"));
        steady.add(new Parameter<double>("relativeTolerance", 	solver->relativeTolerance, "Relative precision of solution components"));
        Add(steady);
    }

    if(!Count())
    {
        Log(lInfo)<<"A model has not been loaded, so  capabilities are not available.";
    }
}

void CapsSupport::Add(const Capability& section)
{
    mCapabilities.push_back(section);
}

string CapsSupport::AsXMLString()
{
    //Create XML
    rrXMLDoc doc;
    xml_node mainNode = doc.append_child("caps");
    mainNode.append_attribute("name") = mName.c_str();
    mainNode.append_attribute("description") = mDescription.c_str();

    //Add sections
    for(int i = 0; i < Count(); i++)
    {
        Capability& section = mCapabilities[i];

        pugi::xml_node section_node = mainNode.append_child("section");
        section_node.append_attribute("name") 			= section.getName().c_str();
        section_node.append_attribute("method")	 		= section.getMethod().c_str();
        section_node.append_attribute("description") 	= section.getDescription().c_str();

        //Add parameters within each section
        for(int j = 0; j < section.nrOfParameters(); j++)
        {
            rr::BaseParameter* cap = const_cast<rr::BaseParameter*>(&(section[j]));
            pugi::xml_node cap_node = mainNode.append_child("cap");
            cap_node.append_attribute("name") 	= cap->getName().c_str();
            cap_node.append_attribute("value") 	= cap->getValueAsString().c_str();
            cap_node.append_attribute("hint") 	= cap->getHint().c_str();
            cap_node.append_attribute("type") 	= cap->getType().c_str();
        }
    }

    stringstream xmlS;
    doc.print(xmlS,"  ", format_indent);
    return xmlS.str();
}


u_int CapsSupport::Count()
{
    return mCapabilities.size();
}

}

