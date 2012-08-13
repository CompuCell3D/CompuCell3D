#ifndef PYSCRIPTRUNNER_H
#define PYSCRIPTRUNNER_H

#include <string>

class SimthreadBase;

//void simulationPython(std::string &scriptName,std::string &path);
void simulationPython(std::string &scriptName,std::string &path ,SimthreadBase* simthreadBasePtr );

class PyScriptRunner{
   public:
      PyScriptRunner(){}
      virtual ~PyScriptRunner(){} 
      virtual void simulationPython(std::string &scriptName,std::string &path){}

};

#endif
