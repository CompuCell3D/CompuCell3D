#ifndef BASICMODULEMANAGER_H
#define BASICMODULEMANAGER_H

#include "BasicException.h"
#include "BasicModuleInfo.h"

#include <map>
#include <list>
#include <string>
//#include <dlfcn.h>
#include <string.h>
#include <iostream>
#include "WindowsGlob.h"
#include <windows.h>
//#include <CenterModel/Components/ForceTerm.h>








template <class T>
class BasicModuleManager {

private:
  //typedefs for module accessor fcn ptrs
  typedef T * (*CREATEMODULE_PTR)(); 
  typedef std::string (*STRINGEXTRACTFCN_PTR)(); 
  typedef int (*INTEXTRACTFCN_PTR)(); 



  //typedef typename std::map<std::string, BasicClassFactoryBase<T> *>
  //factories_t;
  typedef typename std::map<std::string, BasicModuleInfo *> infoMap_t;
  typedef typename std::map<std::string, T *> modules_t;
  typedef typename std::map<std::string, HINSTANCE > libraryHandles_t;
  
  typedef typename std::map<std::string, BasicModuleInfo> nameToBasicModuleInfoMap_t;

  ///// A map from module name to registered factory.
  //factories_t factories;

  /// A map from module name to registered info.
  infoMap_t infoMap;

  /// A map from module name to module instance.
  modules_t modules;

  /// A map of loaded library handles.
  libraryHandles_t libraryHandles;


  //a dictionary of module names to ModuleInfo
  nameToBasicModuleInfoMap_t nameToModuleInfoMap;

  /// Holds exceptions thrown by modules on dlopen.
  BasicException *moduleException;



public:
  /// May need to ifdef this for windows dlls.
  static const char libExtension[];


  BasicModuleManager() :
    moduleException(0)  
    {}

  /** 
   * Destruct the module manager.
   * Deallocates all modules and closes all module libraries.
   */
  virtual ~BasicModuleManager() {
    unload();    
    closeLibraries();
  }
 
  /// Deallocate all modules in the correct order.
  void unload() {
    while (!modules.empty()) {
      if (!modules.begin()->second) modules.erase(modules.begin());
      else{ 
          delete modules.begin()->second;
          modules.begin()->second=0;
          cerr<<"deleted "<<modules.begin()->first<<endl;

      }
          
    }
  }
  
  /// Close all shared library handles.
  void closeLibraries() {
    typename libraryHandles_t::iterator it;
    for (it = libraryHandles.begin(); it != libraryHandles.end(); it++){
      if (it->second) FreeLibrary((HMODULE)it->second);
      cerr<<"closed library "<<it->first<<endl;  
    }
    libraryHandles.clear();
  }

  /**
   * If module1 has not yet been registered this function
   * will return false;
   *
   * @return True if module1 has module2 in its dependency list, false
   *         otherwise.
   */
  bool dependsOn(std::string module1, std::string module2) {
    const BasicModuleInfo *info = infoMap[module1];
    if (!info) return false;

    // Assumes dependency cycles have been disallowed on module registration!
    std::string dep;
    for (unsigned int i = 0; i < info->getNumDeps(); i++) {
      dep = info->getDependency(i);
      if (dep == module2 || dependsOn(dep, module2)) return true;
    }
    return false;
  }

  /// @return True if module with name @param moduleName is loaded.
  bool isLoaded(std::string moduleName) {
    return modules[moduleName] != 0;
  }


  /**
   * Get a pointer to a named module.  If an instance has not yet been
   * allocated it will be created.  If the module name is unknown a
   * BasicException will be thrown.
   *
   * @param moduleName The name of the module.
   *
   * @return A pointer to the named module.
   */


  T *get(const std::string moduleName, bool *_moduleAlreadyRegisteredFlag=0) {

    T *module = modules[moduleName];
    if (module){
      if(_moduleAlreadyRegisteredFlag)
         *_moduleAlreadyRegisteredFlag=true;
      return module;
   }

    nameToBasicModuleInfoMap_t::iterator mitr = nameToModuleInfoMap.find(moduleName);

    if (mitr==nameToModuleInfoMap.end())
        return 0; //module info does not exist

    BasicModuleInfo &info=mitr->second;

	HINSTANCE handle=LoadLibrary(info.fileName.c_str());


    if (handle==NULL)
		THROW(std::string("BasicModuleManagerFlex::get() error:"));

    libraryHandles[info.fileName]=handle;

    //creating module instance
    CREATEMODULE_PTR createModulePtr = (CREATEMODULE_PTR) GetProcAddress(handle, "createModule");    
    if (createModulePtr){
        module=(createModulePtr)();

        cerr<<" CONSTRUCTED "<<info.name<<endl;
        
        if (!module){
            THROW(std::string("BasicModuleManagerFlex::get() Could not create module ")+info.fileName);    
        }
        modules[moduleName]=module;
        cerr<<module->getName()<<endl;

    }

    cerr<<"Library name="<<info.fileName<<endl;

    
	 
      if(_moduleAlreadyRegisteredFlag)
         *_moduleAlreadyRegisteredFlag=false;
    return module;
  }




  /// @return A reference to the named modules info structure.
  const BasicModuleInfo *getModuleInfo(const std::string moduleName) {
    typename infoMap_t::iterator it = infoMap.find(moduleName);
    ASSERT_OR_THROW(std::string("Module '") + moduleName + " not found!",
		    it != infoMap.end());

    return it->second;
  }

  /**
   * @return a map of module information structures .
   */
  infoMap_t &getModuleInfos() {return infoMap;}
  


  /** 
   * Scans all module libraries on a ':' separated path list.
   * 
   * @param path The module library path list.
   */

  void scanLibraries(const std::string path) {
    using namespace std;
    cerr<<"THIS IS PATH="<<path<<endl;
    char *cPath = strdup(path.c_str());

    char *aPath = strtok(cPath, ";");
    cerr<<"THIS IS apath"<<aPath<<endl;
    while (aPath != NULL) {
      scanSingleLibraryPath(aPath);

      aPath = strtok(NULL, ":");
    }

    free(cPath);
  }


  /**
   * Scans (by loading and unloading) a module library.
   *
   * @param filename Path to module library.
   */


  void scanLibrary(const std::string fname) {
	  using namespace std;
    // Get the library name
    std::string fileName;
    std::string::size_type pos = fname.find_last_of("/");
    if (pos != std::string::npos) fileName = fname.substr(pos + 1);
    else fileName = fname;

    
    

	cerr<<"Library Name="<<fname<<endl;

    //void *handle = dlopen(filename.c_str(), RTLD_LAZY | RTLD_GLOBAL);
	//cerr<<"Will try to load the following library: "<<filename<<endl;
	HINSTANCE handle=LoadLibrary(fileName.c_str());
	//cerr<<"I have loaded the library "<<name<<endl;


    if (handle==NULL)
		THROW(std::string("BasicModuleManagerFlex::scanLibrary() error:"));

    
    

    cerr<<"Library name="<<fileName<<endl;
    //char* forceName;
    //forceName = (char* ) GetProcAddress(handle, "forceName");    
    //cerr<<"forceName="<<forceName<<endl;



    BasicModuleInfo moduleInfo;
    moduleInfo.fileName=fileName;

    STRINGEXTRACTFCN_PTR strExtractFcnPtr;
    INTEXTRACTFCN_PTR intExtractorFcnPtr;

    strExtractFcnPtr = (STRINGEXTRACTFCN_PTR) GetProcAddress(handle, "getModuleName");    
    if (strExtractFcnPtr){
        moduleInfo.name=(strExtractFcnPtr)();;

    }

    strExtractFcnPtr = (STRINGEXTRACTFCN_PTR) GetProcAddress(handle, "getModuleType");    
    if (strExtractFcnPtr){
        moduleInfo.type=(strExtractFcnPtr)();
    }


    strExtractFcnPtr = (STRINGEXTRACTFCN_PTR) GetProcAddress(handle, "getModuleAuthor");    
    if (strExtractFcnPtr){        
        moduleInfo.author=(strExtractFcnPtr)();
    }

    intExtractorFcnPtr = (INTEXTRACTFCN_PTR) GetProcAddress(handle, "getVersionMajor");    
    if (intExtractorFcnPtr){        
        moduleInfo.versionMajor=(intExtractorFcnPtr )();
    }

    intExtractorFcnPtr = (INTEXTRACTFCN_PTR) GetProcAddress(handle, "getVersionMinor");    
    if (intExtractorFcnPtr){        
        moduleInfo.versionMinor=(intExtractorFcnPtr )();
    }

    intExtractorFcnPtr = (INTEXTRACTFCN_PTR) GetProcAddress(handle, "getVersionSubMinor");    
    if (intExtractorFcnPtr){        
        moduleInfo.versionSubMinor=(intExtractorFcnPtr )();
    }
    
    nameToModuleInfoMap[moduleInfo.name]=moduleInfo;

    cerr<<"moduleInfo="<<moduleInfo<<endl;

    FreeLibrary((HMODULE)handle);

    if (moduleException) {
      BasicException e = *moduleException;
      delete moduleException;
      throw BasicException(std::string("Exception while loading library '") +
			   fname + "'", e);
    }
  }



  /**
   * Used by statically allocated modules to pass any exceptions back
   * to the library loader.
   */
  void setModuleException(BasicException e) {
    if (!moduleException)
      moduleException = new BasicException(e);
  }



protected:


  modules_t & getModuleMapBPM(){return modules;}

  /**
   * Scans all libraries (by loading and unloading) on a single path.
   *
   * @param path A single path.
   */


  void scanSingleLibraryPath(const std::string path) {
    using namespace std;
    glob_t globbuf;
    // NOTE the change from "/" + "*" UNIX to "*\" WINDOWS
    cerr<<"path to glob="<<path<<endl;
	std::string pathGlob = path + "\\*" + libExtension;

    cerr<<"pathGlob="<<pathGlob<<endl;

    if (glob(pathGlob.c_str(), 0, NULL, &globbuf) == 0) {
      cerr<<"globbuf.gl_pathc="<<globbuf.gl_pathc<<endl;
      for (unsigned int i = 0; i < globbuf.gl_pathc; i++){
	       cerr<<" globbuf.gl_pathv[i]="<<globbuf.gl_pathv[i]<<endl;
      }
      for (unsigned int i = 0; i < globbuf.gl_pathc; i++){
	       scanLibrary(globbuf.gl_pathv[i]);
           
      }
    }

    globfree(&globbuf);
  }

};


template<class T>
const char BasicModuleManager<T>::libExtension[] = ".dll";

#endif
