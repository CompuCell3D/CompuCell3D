#include "PluginInfo.h"
#include <iostream>

using namespace CompuCell3D;

PluginInfo::PluginInfo(const std::string &_name, const std::string _description) :
        name(_name), description(_description) {};

const std::string PluginInfo::getName() const { return name; }

const std::string PluginInfo::getDescription() const { return description; }

const unsigned int PluginInfo::getNumDeps() const { return (unsigned int) deps.size(); }

const std::string PluginInfo::getDependency(const int i) const { return deps[i]; }

void PluginInfo::registerDependency(const std::string &pluginName) {
    for (auto &x: deps) if (x == pluginName) return;
    deps.push_back(pluginName);
}

bool PluginInfo::dependsOn(const std::string &pluginName) {
    for (auto &x: deps) if (x == pluginName) return true;
    return false;
}

std::ostream &operator<<(std::ostream &_os, PluginInfo &_info) {

    _os << _info.getName() << std::string(": ") << _info.getDescription() << std::endl;
    return _os;
}
