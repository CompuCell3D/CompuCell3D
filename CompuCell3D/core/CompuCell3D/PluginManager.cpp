

#include "PluginManager.h"

using namespace CompuCell3D;
using namespace std;

template<typename PluginType>
typename PluginManager<PluginType>::plugins_t &PluginManager<PluginType>::getPluginMap() { return plugins; }

template<typename PluginType>
typename PluginManager<PluginType>::infos_t &PluginManager<PluginType>::getPluginInfos() { return infos_list; }

template<typename PluginType>
std::list <std::string> PluginManager<PluginType>::getLibraryNames() {
    std::list <std::string> o;
    for (auto &x: libHandles) o.push_back(x.first);
    return o;
}

template<typename PluginType>
PluginType *PluginManager<PluginType>::get(const std::string pluginName, bool *_alreadyRegistered) {
    if (plugins[pluginName]) {
        if (_alreadyRegistered)
            *_alreadyRegistered = true;
        return plugins[pluginName];
    }

    // Create the plugin
    PluginType *plugin = getFactory(pluginName)->create();
    init(plugin);

    plugins[pluginName] = plugin;
    proxies[pluginName]->instance = plugin;

    if (_alreadyRegistered) *_alreadyRegistered = false;
    return plugin;
}

template<typename PluginType>
bool PluginManager<PluginType>::isLoaded(const std::string &pluginName) {
    return plugins.find(pluginName) != plugins.end();
}

template<typename PluginType>
bool PluginManager<PluginType>::isRegistered(const std::string &pluginName) {
    return factories.find(pluginName) != factories.end();
}

template<typename PluginType>
void
PluginManager<PluginType>::registerDependency(const std::string &parentPlugin, const std::string &dependentPlugin) {
    if (!isRegistered(parentPlugin)) throw CC3DException(std::string("Parent plugin not registered: ") + parentPlugin);
    if (!isRegistered(dependentPlugin))
        throw CC3DException(std::string("Dependent plugin not registered: ") + dependentPlugin);
    infos[parentPlugin]->registerDependency(dependentPlugin);
}

template<typename PluginType>
bool PluginManager<PluginType>::dependsOn(const std::string &parentPlugin, const std::string &dependentPlugin) {
    if (!isRegistered(parentPlugin)) throw CC3DException(std::string("Parent plugin not registered: ") + parentPlugin);
    if (!isRegistered(dependentPlugin))
        throw CC3DException(std::string("Dependent plugin not registered: ") + dependentPlugin);
    return infos[parentPlugin]->dependsOn(dependentPlugin);
}

template<typename PluginType>
void PluginManager<PluginType>::destroyPlugin(const std::string &pluginName) {
    if (!isLoaded(pluginName)) return;

    PluginInfo *info = infos[pluginName];

    // Deallocate all dependencies
    for (unsigned int x = 0; x < info->getNumDeps(); x++) destroyPlugin(info->getDependency(x));

    // Deallocate and remove the plugin
    proxies[pluginName]->instance = NULL;
    auto itr = plugins.find(pluginName);
    getFactory(pluginName)->destroy(itr->second);
    plugins.erase(itr);
}

template<typename PluginType>
void PluginManager<PluginType>::unload() {
    while (!plugins.empty()) destroyPlugin(plugins.begin()->first);
}

template<typename PluginType>
void PluginManager<PluginType>::loadLibrary(const std::string &filename) {
    // Get name
    auto pos = filename.find_last_of("/");
    std::string libName;
    if (pos != std::string::npos) libName = filename.substr(pos + 1);
    else libName = filename;

    if (libHandles[libName]) return;

    // Load it!

#ifdef CC3D_ISWIN
    libHandle_t handle = LoadLibrary(filename.c_str());
#else
    libHandle_t handle = dlopen(filename.c_str(), RTLD_LAZY | RTLD_GLOBAL);
#endif
//    cerr<<"handle="<<handle<<endl;
    if (!handle) throw CC3DException(std::string("Load library error: ") + filename);

    libHandles[libName] = handle;
}

template<typename PluginType>
void PluginManager<PluginType>::loadLibraryFromPath(const std::string &path) {
    for (std::string f: filesInDir(path, "*" + libExtension)) {
//        cerr<<"loading "<<f<<endl;
        loadLibrary(f);
    }
}

template<typename PluginType>
void PluginManager<PluginType>::loadLibraries(const std::string path) {
    for (auto &s: splitString(path, pathDelim)){
//        cerr<<"module path "<<path<<endl;
        try {
            loadLibraryFromPath(s);
        }catch (std::exception & e){
            cerr<<"Error loading module: "<<path<<endl;
            cerr<<"exception "<<e.what()<<endl;
        }
    }

}

template<typename PluginType>
void PluginManager<PluginType>::closeLibrary(const std::string &libName) {
    if (!libHandles[libName]) return;

#ifdef CC3D_ISWIN
    bool result = FreeLibrary((HMODULE)libHandles[libName]) != 0;
#else
    bool result = dlclose(libHandles[libName]) == NULL;
#endif

    if (!result) throw CC3DException(std::string("Close library error: ") + libName);
}

template<typename PluginType>
void PluginManager<PluginType>::closeLibraries() {
    for (auto x: libHandles) closeLibrary(x.first);
    libHandles.clear();
}

template<typename PluginType>
typename PluginManager<PluginType>::proxy_t *PluginManager<PluginType>::registerPlugin(PluginInfo *info,
                                                                                       typename PluginManager<PluginType>::PluginFactory *factory) {
    // Validate unique name
    if (isRegistered(info->getName()))
        throw CC3DException(std::string("Plugin name is not unique: ") + info->getName());

    // Register
    infos[info->getName()] = info;
    infos_list.push_back(info);
    factories[info->getName()] = factory;
    PluginManager<PluginType>::proxy_t *proxy = new PluginManager<PluginType>::proxy_t();
    proxies[info->getName()] = proxy;
    return proxy;
}

template<typename PluginType>
typename PluginManager<PluginType>::PluginFactory *
PluginManager<PluginType>::getFactory(const std::string &pluginName) {
    if (!isRegistered(pluginName)) throw CC3DException(std::string("Plugin not registered: ") + pluginName);
    return factories[pluginName];
}


template
class CompuCell3D::PluginProxy<Plugin>;

template
class CompuCell3D::PluginProxy<Steppable>;

template
class CompuCell3D::PluginProxy<PluginBase>;

template
class CompuCell3D::PluginManager<Plugin>;

template
class CompuCell3D::PluginManager<Steppable>;

template
class CompuCell3D::PluginManager<PluginBase>;
