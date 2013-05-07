#ifndef rrSimulationSettingsH
#define rrSimulationSettingsH
#include "rrObject.h"
#include "rrStringList.h"
//---------------------------------------------------------------------------

namespace rr
{
using std::vector;

class RR_DECLSPEC SimulationSettings : public rrObject
{
    protected:
        void                    ClearSettings();
    public:
                                SimulationSettings();

        int                     mSteps;
        double                  mStartTime;
        double                  mDuration;
        double                  mEndTime;
        double                  mAbsolute;
        double                  mRelative;
        StringList              mVariables;
        StringList              mAmount;
        StringList              mConcentration;
        bool                    LoadFromFile(const string& fName);
};


/*    Field             Explanation   (from http://sbml.org/Software/SBML_Test_Suite/Case_Descriptions)
    start:              The start time of the simulation time-series data in the output (CSV) file. Often this is 0, but not necessarily.

    duration:           The duration of the simulation run, in seconds.

    steps:              The number of steps at which the output is sampled. The samples are evenly spaced.

                        When a simulation system calculates the data points to record, it will typically divide the duration by the number of time steps.
                        Thus, for X steps, the data file will have X+1 data rows.
    variables:          The variables (in addition to time) whose values are tabulated in the CSV file. These are SBML model id's separated by commas.
                        Order is significant. Important: if a symbol in this list refers to a species in the model, then that symbol will
                        also be listed in either the amount or concentration lists below.

    absolute:           A float-point number representing the absolute difference permitted for this test case when evaluating a software tool.
                        The formula used to calculate data point differences is discussed below.

    relative:           A float-point number representing the relative difference permitted for this test case when evaluating a software tool.
                        The value of 0.0001 was the tolerance agreed upon by the SBML community during discussions at SBML Hackathons in 2008.
                        The formula used to calculate data point differences is discussed below.

    amount:             A list of the variable whose output in the results file is in amount (not concentration) units.
                        This list of variables must be a subset of the names listed in variables.

    concentration:      A list of the variable whose output in the results file is in concentration (not amount) units.
                        This list of variables must be a subset of the names listed in variables.
*/
} //End of namespace
#endif
