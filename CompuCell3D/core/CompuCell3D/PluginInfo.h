#ifndef PLUGININFO_H
#define PLUGININFO_H

#include <string>
#include <vector>

#include "CompuCellLibDLLSpecifier.h"

namespace CompuCell3D {

    /**
	Written by T.J. Sego, Ph.D.
	*/

    // Simple information about a plugin
    class COMPUCELLLIB_EXPORT PluginInfo {

        std::string name;
        std::string description;
        unsigned int nDeps;
        std::vector <std::string> deps;

    public:

        PluginInfo(const std::string &_name, const std::string _description);

        const std::string getName() const;

        const std::string getDescription() const;

        const unsigned int getNumDeps() const;

        const std::string getDependency(const int i) const;

        // Registers a plugin as a dependency of this plugin
        void registerDependency(const std::string &pluginName);

        // Tests whether a plugin is a dependency
        bool dependsOn(const std::string &pluginName);

        friend std::ostream &operator<<(std::ostream &_os, PluginInfo &_info);

    };

    std::ostream &operator<<(std::ostream &_os, PluginInfo &_info);
};

#endif