#include <Python.h> 
#include <cstdio>
#include <iostream>
#include <sstream>
#include <PyScriptRunner.h>
using namespace std;


void simulationPython(std::string &scriptName, std::string &path ,SimthreadBase* simthreadBasePtr ){
	//FILE *fp=fopen(scriptName.c_str(),"r");

    ostringstream simthreadPtrStream;
    cerr<<"FROM SIMULAITON PYTHON simthreadBasePtr="<<simthreadBasePtr<<endl;
    simthreadPtrStream<<"simthreadPtr="<<(long)simthreadBasePtr;
    cerr<<"THIS IS PYTHON SCRITP "<<simthreadPtrStream.str()<<endl;

   ostringstream scriptStream;
   //scriptStream<<"execfile(\'"<<scriptName<<"\')";

	std::string scriptnameLocal="pythonSetupScripts/CompuCellPythonSimulation.py";
	scriptStream<<"execfile(\'"<<scriptnameLocal<<"\')";
   
    

   //Py_Initialize();
   PyThreadState* interpreter=Py_NewInterpreter();
   PyRun_SimpleString(simthreadPtrStream.str().c_str());
   PyRun_SimpleString(scriptStream.str().c_str());
   //PyRun_SimpleFile(fp,scriptName.c_str()); //will not work under windows
   cerr<<"BEFORE Py_Finalize()"<<endl;
   Py_EndInterpreter(interpreter);
   //Py_Finalize();
   cerr<<"AFTER Py_Finalize()"<<endl;   
//fclose(fp);

//     cerr<<" THIS IS PY_IsINITIALIZED="<<Py_IsInitialized()<<endl;
//     ostringstream simthreadPtrStream1;
//     cerr<<"FROM SIMULAITON PYTHON simthreadBasePtr1="<<simthreadBasePtr<<endl;
//     simthreadPtrStream1<<"simthreadPtr="<<(long)simthreadBasePtr;
//     cerr<<"THIS IS PYTHON SCRITP1 "<<simthreadPtrStream.str()<<endl;
// 
//    ostringstream scriptStream1;
//    scriptStream1<<"execfile(\'"<<"bacterium_macrophage-player-new-syntax-steering.py"<<"\')";
//    
// 
// //    Py_Initialize();
//    PyThreadState* interpreter1=Py_NewInterpreter();
//    PyRun_SimpleString(simthreadPtrStream.str().c_str());
//    PyRun_SimpleString(scriptStream1.str().c_str());
//    //PyRun_SimpleFile(fp,scriptName.c_str()); //will not work under windows
//    cerr<<"BEFORE Py_Finalize()1"<<endl;
//    Py_EndInterpreter(interpreter1);
// //    Py_Finalize();
//    cerr<<"AFTER Py_Finalize()1"<<endl;   



}


//void simulationPython(std::string &scriptName, std::string &path){
//   FILE *fp=fopen(scriptName.c_str(),"r");
//   
//   Py_Initialize();
//   PyRun_SimpleFile(fp,scriptName.c_str());
//   Py_Finalize();
//   fclose(fp);
//
//
//}
