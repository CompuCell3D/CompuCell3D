#ifndef CommandLineParametersH
#define CommandLineParametersH
#include <string>
#include "rrLogger.h"
//---------------------------------------------------------------------------

using std::string;
using namespace rr;
string Usage(const string& prg);
class Args
{
    public:
                                        Args();
        virtual                        ~Args(){}
        LogLevel                        CurrentLogLevel;                            //option v:
        string                          ModelFileName;                              //option m:
        bool                            SaveResultToFile;                           //option f
        string                          InstallFolder;                              //option i:
        string                          TempDataFolder;                             //option t:
        bool                            Pause;                                      //option p
        double                          StartTime;                                  //option s
        double                          Duration;
        double                          EndTime;                                    //option e
        int                             Steps;                                      //option z
        string                          SelectionList;      	                    //option l:
        bool							CalculateSteadyState; 	                    //option x
		bool							ComputeAndAssignConservationLaws;			//option y
};

#endif
