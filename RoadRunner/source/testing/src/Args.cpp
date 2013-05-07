#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <sstream>
#include <iomanip>
#include "Args.h"
#include "rrUtils.h"
//---------------------------------------------------------------------------
using namespace std;
Args::Args()
:
SBMLModelsFilePath(""),
ResultOutputFile(""),
TempDataFolder("."),
//DataOutputFolder("."),
//Compiler(""),
//SupportCodeFolder(""),
EnableLogging(false)
{}

string Usage(const string& prg)
{
    stringstream usage;
    usage << "\nUSAGE for "<<prg<<"\n\n";
    usage<<left;
    usage<<setfill('.');
    usage<<setw(25)<<"-m<FilePath>"                 <<" Folder where the program will read sbml models. \n";
	usage<<setw(25)<<"-l<FilePath>"                 <<" Compiler \n";
	usage<<setw(25)<<"-s<FilePath>"                 <<" Support code folder location\n";
	usage<<setw(25)<<"-r<FilePath>"                 <<" Results output file, with path.\n";
    usage<<setw(25)<<"-t<FilePath>"                 <<" Temporary data output folder. If not given, temp files are output to current directory\n";
    usage<<setw(25)<<"-d<FilePath>"                 <<" Data outputFolder\n";
    usage<<setw(25)<<"-v"                 			<<" Enable Logging\n";
    usage<<setw(25)<<"-?"                           <<" Shows the help screen.\n\n";

	usage<<"\nSystems Biology, UW 2012\n";
	rr::Pause();
    return usage.str();
}

