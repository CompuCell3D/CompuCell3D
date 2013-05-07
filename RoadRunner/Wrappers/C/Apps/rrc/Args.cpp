#pragma hdrstop
#include <iomanip>
#include "Args.h"
//---------------------------------------------------------------------------

Args::Args()
:
Pause(false),
CurrentLogLevel(lInfo),
ModelFileName(""),
InstallFolder(""),
TempDataFolder("."),
SelectionList(""),
StartTime(0),
Duration(5),
EndTime(StartTime + Duration),
Steps(50),
ComputeAndAssignConservationLaws(true),
CalculateSteadyState(false),
SaveResultToFile(false)
{}

string Usage(const string& prg)
{
    stringstream usage;
    usage << "\nUSAGE for "<<prg<<"\n\n";
    usage<<left;
    usage<<setfill('.');
    usage<<setw(25)<<"-v<debug level>"              <<" Debug levels: Error, Warning, Info, Debug, Debug'n', where n is 1-7. Defualt: Info\n";
    usage<<setw(25)<<"-m<FileName>"                 <<" SBML Model File Name (with path)\n";
    usage<<setw(25)<<"-f"                           <<" Save result to file (file name: \"<modelName>.csv\". If -f is not given, data is output to screen\n";
    usage<<setw(25)<<"-i<FilePath>"                 <<" Install folder. If not given, a pre-defined default or calcuated value is used\n";
    usage<<setw(25)<<"-t<FilePath>"                 <<" Temporary data output folder. If not given, temporary files are output to current directory\n";
    usage<<setw(25)<<"-p"                           <<" Pause before exiting.\n";   
    usage<<setw(25)<<"-s<#>"                        <<" Set the start time for simulation. Default: 0\n";
    usage<<setw(25)<<"-e<#>"                        <<" Set the end time for simulation. Default: 5\n";
    usage<<setw(25)<<"-z<#>"                        <<" Set number of steps in the simulation. Default: 50\n";
    usage<<setw(25)<<"-l<List>"                     <<" Set selection list. Separate variables using ',' or space\n";
    usage<<setw(25)<<"-x"                     		<<" Calculate steady state\n";
    usage<<setw(25)<<"-y"                     		<<" Disable ComputeAndAssignConservationLaws\n";
    usage<<setw(25)<<"-?"                           <<" Shows the help screen.\n\n";

    usage<<"\nSystems Biology, UW 2012\n";
    return usage.str();
}

