//---------------------------------------------------------------------------
#pragma hdrstop
#include <sstream>
#include "Poco/SharedLibrary.h"
#include "rrPluginManager.h"
#include "rrPlugin.h"
#include "rrUtils.h"
#include "rrException.h"

using std::stringstream;
using std::pair;
using Poco::SharedLibrary;

namespace rr
{

bool destroyRRPlugin(Plugin *plugin);

PluginManager::PluginManager(const std::string& folder, const bool& autoLoad, RoadRunner* aRR)
:
mPluginFolder(folder),
mRR(aRR)
{
    if(autoLoad)
    {
        load();
    }
}

PluginManager::~PluginManager()
{}

void PluginManager::setRoadRunnerInstance(RoadRunner* aRR)
{
	mRR = aRR;
}

bool PluginManager::setPluginFolder(const string& dir)
{
	return false;
}

string PluginManager::getPluginFolder()
{
	return mPluginFolder;
}

Plugin*	PluginManager::operator[](const int& i)
{
	if(i >= 0 && i < mPlugins.size())
    {

    	return mPlugins[i].second;
    }
    else
    {
    	return NULL;
    }
}

typedef Plugin* (*createRRPluginFunc)(RoadRunner*);
typedef bool    (*destroyRRPluginFunc)(Plugin* );

bool PluginManager::load()
{
	bool result = false;
    //Throw if plugin folder don't exist
    if(!FolderExists(mPluginFolder))
    {
        throw Exception("Plugin folder do not exists");
    }
    //Look for shared libraries in this folder

//    for(int i = 0; i < nrOfLibs; i++)
    {
    	//Load and create the plugins
	    result = loadPlugin("TestPlugin.dll");
	    result = loadPlugin("fit_one_parameter.dll");
    }
    return result;
}

bool PluginManager::loadPlugin(const string& sharedLib)
{
	try
    {
        SharedLibrary *aLib = new SharedLibrary;
        aLib->load(JoinPath(mPluginFolder, sharedLib));

        //Validate the plugin
        if(aLib->hasSymbol("createRRPlugin"))
        {
            createRRPluginFunc create = (createRRPluginFunc) aLib->getSymbol("createRRPlugin");
            //This plugin
            Plugin* aPlugin = create(mRR);
            if(aPlugin)
            {
                pair< Poco::SharedLibrary*, Plugin* > storeMe(aLib, aPlugin);
                mPlugins.push_back( storeMe );
            }
        }
        else
        {
            //Log some warnings about a bad plugin...?
        }
        return true;
    }
    catch(const Exception& e)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<e.what()<<endl;
		Exception ex(msg.str());
    	throw ex;
    }
    catch(const Poco::Exception& ex)
    {
		stringstream msg;
    	msg<<"Poco exception: "<<ex.displayText()<<endl;
    	Exception newMsg(msg.str());
    	throw newMsg;
    }
}

bool PluginManager::unload()
{
	bool result(true);
    int nrPlugins = getNumberOfPlugins();
	for(int i = 0; i < nrPlugins; i++)
    {
    	pair< Poco::SharedLibrary*, Plugin* >  *aPluginLib = &(mPlugins[i]);
        if(aPluginLib)
        {
            SharedLibrary *aLib 	= aPluginLib->first;
            Plugin*		   aPlugin 	= aPluginLib->second;

            destroyRRPlugin(aPlugin);

            //Then unload
			if(aLib)
			{
				aLib->unload();
			}
            //And remove from container
            aPluginLib->first = NULL;
            aPluginLib->second = NULL;
        }
    }

    //Remove all from container...
    mPlugins.clear();
    return result;
}

StringList PluginManager::getPluginNames()
{
	StringList names;

    int nrPlugins = getNumberOfPlugins();
	for(int i = 0; i < nrPlugins; i++)
    {
    	pair< Poco::SharedLibrary*, Plugin* >  *aPluginLib = &(mPlugins[i]);
        if(aPluginLib)
        {
            Plugin*		   aPlugin 	= aPluginLib->second;

            //Then unload
            names.Add(aPlugin->getName());
        }
    }
    return names;
}

int	PluginManager::getNumberOfPlugins()
{
	return mPlugins.size();
}

int PluginManager::getNumberOfCategories()
{
	return -1;
}

Plugin*	PluginManager::getPlugin(const int& i)
{
	return (*this)[i];
}

Plugin*	PluginManager::getPlugin(const string& name)
{
	for(int i = 0; i < getNumberOfPlugins(); i++)
    {
    	pair< Poco::SharedLibrary*, Plugin* >  aPluginLib = mPlugins[i];
        if(aPluginLib.first && aPluginLib.second)
        {
			Plugin* aPlugin = (Plugin*) aPluginLib.second;
            if(aPlugin && aPlugin->getName() == name)
            {
               	return aPlugin;
            }
        }
    }
    return NULL;
}

// Plugin cleanup function
bool destroyRRPlugin(rr::Plugin *plugin)
{
	//we allocated in the factory with new, delete the passed object
    try
    {
    	delete plugin;
    	return true;
    }
    catch(...)
    {
    	//Bad stuff!
        clog<<"Failed deleting RoadRunner plugin..";
        return false;
    }
}

}

