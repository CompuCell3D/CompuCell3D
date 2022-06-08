

#ifndef PLUGIN_H
#define PLUGIN_H

#include "SimObject.h"

namespace CompuCell3D {
    class Plugin : public SimObject {
    public:
        Plugin() {}

        virtual std::string toString() { return "Plugin"; }

        virtual ~Plugin() {}
    };
};
#endif
