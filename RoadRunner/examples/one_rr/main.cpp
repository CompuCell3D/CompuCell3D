#include <iostream>
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrLogger.h"
using namespace rr;

int main(int argc, char** argv)
{
// 	const char* rootPath = "d:\\Program Files (x86)\\RoadRunner";
        const char* rootPath = "/home/m/install_projects/RR";

	try
    {
        LogOutput::mLogToConsole = true;
        gLog.SetCutOffLogLevel(lInfo);
		string tmpFolder = JoinPath(rootPath, "temp");

        const string modelFile = JoinPath(rootPath, "models", "test_1.xml");
		const string compilerSupportDir = JoinPath(rootPath, "rr_support");
// 		const string compilerFileName = JoinPath(rootPath, "compilers","tcc","tcc.exe");
                
                const string compilerFileName = JoinPath("/usr/bin", "gcc");
                
		cerr<<"modelFile="<<modelFile<<endl;
        //Load modelFiles..
        Log(lInfo)<<" ---------- LOADING/GENERATING MODELS ------";

        RoadRunner rr1(tmpFolder,compilerSupportDir,compilerFileName);
		//Compiler* comp1=rr1.getCompiler();
		//comp1->setCompiler(compilerFileName );

		RoadRunner rr2(tmpFolder,compilerSupportDir,compilerFileName);
		//Compiler* comp2=rr2.getCompiler();
		//comp2->setCompiler(compilerFileName);

		cerr<<"THIS IS BEFORE LOADING SBML"<<endl;

        if(!rr1.loadSBMLFromFile(modelFile, true))
        {
			cerr<<"There was a problem loading model in file: "<<modelFile<<endl;
			Log(lError)<<"There was a problem loading model in file: "<<modelFile;
            throw(Exception("Bad things in loadSBMLFromFile function"));
        }

		cerr<<"THIS IS AFTER LOADING SBML"<<endl;

  //////  	SModelData data;
	 //////   clog<<"Size: "<<sizeof(SModelData)<<endl;
	 //////   clog<<"Size ptr: "<<sizeof(data.eventDelays)<<endl;
		//////cerr<<" SIZE OF ROAD RUNNER="<<sizeof(RoadRunner)<<endl;
  //////      rr1.getModel()->cInitModelData(&data);
  //////      Log(lInfo)<<" ---------- SIMULATE ---------------------";
  //////      Log(lInfo)<<"Data:"<<rr1.simulate();

		//we will skip compilation here assuming that first loadSBMLFromFile got it right
        if(!rr2.loadSBMLFromFile(modelFile, false))
        { 
			cerr<<"There was a problem loading model in file: "<<modelFile<<endl;
			Log(lError)<<"There was a problem loading model in file: "<<modelFile;
            throw(Exception("Bad things in loadSBMLFromFile function"));
        }



  //  	SModelData data;
		//rr1.getModel()->cInitModelData(&data);

  //  	SModelData data2;
		//rr2.getModel()->cInitModelData(&data2);


		double stepSize=0.5;
		int numSteps=20;
		double t;

		rr1.setNumPoints(1);
		for (int i = 0 ; i <numSteps ; ++i){
			t=i*stepSize;
			rr1.setTimeStart(t);
			rr1.setTimeEnd(t+stepSize);
			//Log(lInfo)<<"Data:"<<rr1.simulate();
			rr1.simulate();
			cerr<<"rr1 t="<<t<<" S1="<<rr1.getValue("S1")<<" S2="<<rr1.getValue("S2")<<endl;

			rr2.setTimeStart(t);
			rr2.setTimeEnd(t+stepSize);
			//Log(lInfo)<<"Data:"<<rr1.simulate();
			rr2.simulate();
			cerr<<"rr2 t="<<t<<" S1="<<rr2.getValue("S1")<<" S2="<<rr2.getValue("S2")<<endl;

		}
		
		////////cerr<<"\n\n\n\n will create new RoadRunner instance using precompiled code"<<endl;
  //////      RoadRunner rr2(tmpFolder);
		////////cerr<<"THIS IS BEFORE LOADING SBML"<<endl;


		//////rr1.setNumPoints(1);
		//////for (int i = 0 ; i <numSteps ; ++i){
		//////	t=i*stepSize;
		//////	rr2.setTimeStart(t);
		//////	rr2.setTimeEnd(t+stepSize);
		//////	//Log(lInfo)<<"Data:"<<rr1.simulate();
		//////	rr2.simulate();
		//////	cerr<<"t="<<t<<" S1="<<rr2.getValue("S1")<<" S2="<<rr2.getValue("S2")<<endl;
		//////}


	 //   //clog<<"Size: "<<sizeof(SModelData)<<endl;
	 //   //clog<<"Size ptr: "<<sizeof(data.eventDelays)<<endl;
		////cerr<<" SIZE OF ROAD RUNNER="<<sizeof(RoadRunner)<<endl;
		//rr1.setNumPoints(1);
		//rr1.setTimeStart(0.0);
		//rr1.setTimeEnd(0.5);

		//
  //      
  //      Log(lInfo)<<" ---------- SIMULATE ---------------------";
  //      Log(lInfo)<<"Data:"<<rr1.simulate();

		//rr1.setNumPoints(1);
		//rr1.setTimeStart(0.5);
		//rr1.setTimeEnd(1.0);
		//Log(lInfo)<<"Data:"<<rr1.simulate();


    }
    catch(const Exception& ex)
    {
    	Log(lError)<<"There was a  problem: "<<ex.getMessage();
    }

//    Pause(true);
    return 0;
}

#if defined(CG_IDE)
#pragma comment(lib, "roadrunner-static.lib")
#pragma comment(lib, "poco_foundation-static.lib")
#pragma comment(lib, "rr-libstruct-static.lib")
//#pragma comment(lib, "cprts.lib")
void ResolveUnresolvedExternals()
{
locale loc;
std::ostream st;
st.exceptions(2);
st.precision(1);
st.setf( std::ios::showbase,  std::ios::showbase);
st.eof();
st.bad();
st.operator void *();
st.operator !();

st<<use_facet <codecvt<wchar_t, char, mbstate_t> >(loc).max_length();

char_traits<wchar_t>::char_type str1[25] = L"The Hell Boy";
char_traits<wchar_t>::char_type str2[25] = L"Something To ponder";
char_traits<wchar_t>::move(str2, str1, 10);
char_traits<wchar_t>::copy(str1, str2, 2);
char_traits<wchar_t>::assign(*str1, *str2);
char_traits<wchar_t>::assign(str1, 2, *str2);
char_traits<char>::compare("test", "test", 2);
char_traits<char>::find("test", 2, '2');

//ios_base::precision(2);
double test1 = std::numeric_limits<unsigned int>::quiet_NaN();
double test2 = std::numeric_limits<int>::quiet_NaN();
double test3 = std::numeric_limits<unsigned int>::quiet_NaN();
double test4 = std::numeric_limits<int>::quiet_NaN();
double test5 = std::numeric_limits<double>::quiet_NaN();
double test6 = std::numeric_limits<float>::quiet_NaN();
double test7 = std::numeric_limits<float>::infinity();
double test8 = std::numeric_limits<double>::infinity();
double test9 = std::numeric_limits<int>::max();
}

#endif


