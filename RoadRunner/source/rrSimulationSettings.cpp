#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <map>
#include "rrLogger.h"
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "rrSimulationSettings.h"
//---------------------------------------------------------------------------
using namespace std;
namespace rr
{

SimulationSettings::SimulationSettings()
:
mSteps(50),
mStartTime(0),
mDuration(5),
mEndTime(mStartTime + mDuration),
mAbsolute(1.e-7),
mRelative(1.e-4)
{}

void SimulationSettings::ClearSettings()
{
    mSteps      = 0;
    mStartTime  = 0;
    mDuration   = 0;
    mVariables.clear();
    mAmount.clear();
    mConcentration.clear();
}

bool SimulationSettings::LoadFromFile(const string& _FName)
{
    string fName(_FName);

    if(!fName.size())
    {
        Log(lError)<<"Empty file name for setings file";
        return false;
    }
    else
    {
        ClearSettings();
        map<string, string> settings;
        map<string, string>::iterator it;
        //Read each line in the settings file
        vector<string> lines = GetLinesInFile(fName);
        for(int i = 0; i < lines.size(); i++)
        {
            vector<string> line = SplitString(lines[i], ":");
            if(line.size() == 2)
            {
                settings.insert( pair<string, string>(line[0], line[1]));
            }
            else
            {
                Log(lDebug2)<<"Empty line in settings file: "<<lines[i];
            }
        }

        Log(lDebug3)<<"Settings File =============";
        for (it = settings.begin() ; it != settings.end(); it++ )
        {
            Log(lDebug) << (*it).first << " => " << (*it).second;
        }
        Log(lDebug)<<"===========================";

        //Assign values
        it = settings.find("start");
        mStartTime = (it != settings.end())   ? ToDouble((*it).second) : 0;

        it = settings.find("duration");
        mDuration = (it != settings.end())    ? ToDouble((*it).second) : 0;

        it = settings.find("steps");
        mSteps = (it != settings.end())       ? ToInt((*it).second) : 50;

        it = settings.find("absolute");
        mAbsolute = (it != settings.end())    ? ToDouble((*it).second) : 1.e-7;

        it = settings.find("relative");
        mRelative = (it != settings.end())    ? ToDouble((*it).second) : 1.e-4;

        mEndTime = mStartTime + mDuration;

        it = settings.find("variables");
        if(it != settings.end())
        {
            vector<string> vars = SplitString((*it).second, ",");
            for(int i=0; i < vars.size(); i++)
            {
                mVariables.push_back(Trim(vars[i]));
            }
        }

        it = settings.find("amount");
        if(it != settings.end())
        {
            vector<string> vars = SplitString((*it).second, ",");
            for(int i=0; i < vars.size(); i++)
            {
                string rec = Trim(vars[i]);
                if(rec.size())
                {
                    mAmount.push_back(rec);
                }
            }
        }

        it = settings.find("concentration");
        if(it != settings.end())
        {
            vector<string> vars = SplitString((*it).second, ",");
            for(int i=0; i < vars.size(); i++)
            {
                string rec = Trim(vars[i]);
                if(rec.size())
                {
                    mConcentration.push_back(rec);
                }
            }
        }
    }

//    if(mEngine)
//    {
//        mEngine->useSimulationSettings(mSettings);
//
//        //This one creates the list of what we will look at in the result
//        mEngine->CreateSelectionList();
//    }

    return true;

}

}
