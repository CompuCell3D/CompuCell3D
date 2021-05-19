/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

#ifndef PLUGINMANAGER_H
#define PLUGINMANAGER_H

#ifndef CC3D_ISWIN

#if defined(_WIN32) || defined(__MINGW32__)
#define CC3D_ISWIN
#endif // defined(_WIN32) || defined(__MINGW32__)

#endif // CC3D_ISWIN

#if defined(CC3D_ISWIN)

#include <windows.h>
typedef typename HINSTANCE libHandle_t;

#else

#include <dlfcn.h>
typedef typename void* libHandle_t;

#endif // defined(CC3D_ISWIN)

#include <iostream>
#include <list>
#include <map>
#include <vector>

#include <PublicUtilities/FileUtils.h>
#include <PublicUtilities/StringUtils.h>
#include "CC3DExceptions.h"
#include "ExtraMembers.h"
#include "PluginInfo.h"

#include "PluginBase.h"
#include "Plugin.h"
#include "Steppable.h"

#include "CompuCellLibDLLSpecifier.h"

namespace CompuCell3D {
  
  class Simulator;
  class PluginInfo;
  
  // Proxy of plugin; proxies have a reference to its plugin instance when it is alive. 
  template<typename PluginType>
  class COMPUCELLLIB_EXPORT PluginProxy {

  public:
  
    // Pointer to instance when instance is alive, and otherwise NULL. 
    PluginType* instance;
  };


  // Assumes plugins resolve their own dependencies through API
  template<typename PluginType>
  class COMPUCELLLIB_EXPORT PluginManager {

     Simulator *simulator;

  public:

    typedef typename std::map<std::string, PluginType *> plugins_t;
    typedef typename ExtraMembersFactory<PluginType> PluginFactory;
    typedef typename PluginProxy<PluginType> proxy_t;
    typedef typename std::list<PluginInfo* > infos_t;

  private:
    
    typedef typename std::map<std::string, PluginInfo* > pluginInfo_t;
    typedef typename std::map<std::string, PluginFactory* > pluginFactory_t;
    typedef typename std::map<std::string, libHandle_t> libraryHandles_t;
    typedef typename std::map<std::string, proxy_t* > proxies_t;

    plugins_t plugins;
    pluginInfo_t infos;
    pluginFactory_t factories;
    libraryHandles_t libHandles;
    proxies_t proxies;

#ifdef CC3D_ISWIN
    char* pathDelim = ";";
    std::string libExtension = ".dll";
#else
    char* pathDelim = ":";
    std::string libExtension = ".so";
#endif

  public:
    
    PluginManager() : simulator(0) {}
    
    virtual ~PluginManager() {}
    
    // Returns map of plugins by name
    plugins_t & getPluginMap();

    // Returns info for all registered plugins
    infos_t getPluginInfos();

    // Returns name of all loaded libraries
    std::list<std::string> getLibraryNames();

    // Sets the simulator
    void setSimulator(Simulator *simulator) {this->simulator = simulator;}

    // Initializes a plugin
    virtual void init(PluginType *plugin) {
      if (!simulator) throw CC3DException("PluginManager::init() Simulator not set!");
      //plugin->init(simulator);
    }

    // Gets a plugin by name; plugin is created if necessary; throws a CC3DException is 
    // the plugin name is unknown. 
    PluginType* get(const std::string pluginName, bool* _alreadyRegistered = 0);

    // Tests whether a plugin is loaded by name
    bool isLoaded(const std::string& pluginName);

    // Tests whether a plugin is registered by name
    bool isRegistered(const std::string& pluginName);

    // Registers a plugin as a dependency; throws a CC3DException if either plugin is not registered. 
    void registerDependency(const std::string& parentPlugin, const std::string& dependentPlugin);

    // Tests whether a one plugin depends on another; throws a CC3DException if either plugin is not registered. 
    bool dependsOn(const std::string& parentPlugin, const std::string& dependentPlugin);

    // Deallocates a plugin
    void destroyPlugin(const std::string& pluginName);

    // Deallocates all plugins
    void unload();

    // Loads a library from file; load is ignored if library is already loaded. 
    void loadLibrary(const std::string& filename);

    // Loads all libraries at a path
    void loadLibraryFromPath(const std::string& path);

    // Loads all libraries specified on paths specified in a delimited string
    void loadLibraries(const std::string path);

    // Closes a loaded library by library name
    void closeLibrary(const std::string& libName);

    // Closes all libraries
    void closeLibraries();

    // Registers a plugin
    proxy_t* registerPlugin(PluginInfo* info, PluginFactory* factory);

  protected:

    // Simple factory accessor that throws a CC3DException when a plugin factory is not found by plugin name
    PluginFactory* getFactory(const std::string& pluginName);

  };

  /**
   * @brief Main entry point for registering plugins and steppables
   * 
   * @tparam PluginType base type of plugin
   * @tparam ImplType implementation type of plugin
   * @param pluginName name of plugin
   * @param pluginDescription description of plugin
   * @param manager manager to which the plugin should be registered
   * @return plugin proxy
   */
  template<class PluginType, class ImplType>
  PluginProxy<PluginType>* registerPlugin(const std::string& pluginName, const std::string& pluginDescription, PluginManager<PluginType>* manager) {
    return manager->registerPlugin(new PluginInfo(pluginName, pluginDescription), new ExtraMembersCastor<PluginType, ImplType>());
  }

};
#endif
