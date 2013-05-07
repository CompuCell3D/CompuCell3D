#ifndef rrPluginManagerH
#define rrPluginManagerH
#include <vector>
#include "rrObject.h"
#include "rrStringList.h"
//---------------------------------------------------------------------------
/* A minimalistic Plugin manager. */
namespace rr
{

//Abstract class for plugins
class RoadRunner;
class Plugin;

class RR_DECLSPEC PluginManager : public rrObject
{
	private:
        string			   			mPluginFolder;
        vector< pair< Poco::SharedLibrary*, Plugin* > >
        					 		mPlugins;
        RoadRunner		   *mRR;		//This is a handle to the roadRunner instance, creating the pluginManager

    public:
	    				           	PluginManager(const std::string& pluginFolder = gEmptyString, const bool& autoLoad = false, RoadRunner* aRR = NULL);
        				           ~PluginManager();
		bool			           	setPluginFolder(const string& dir);
		string			           	getPluginFolder();
		bool 			           	load();
		bool 			           	loadPlugin(const string& sharedLib);
		bool 			           	unload();
        int				           	getNumberOfPlugins();
		int                         getNumberOfCategories();
        Plugin*			           	getPlugin(const int& i);
        Plugin*			           	getPlugin(const string& name);
        Plugin*	   					operator[](const int& i);
        void						setRoadRunnerInstance(RoadRunner* aRR);
        StringList					getPluginNames();

};
}


#endif
