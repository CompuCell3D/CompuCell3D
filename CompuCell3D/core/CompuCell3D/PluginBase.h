#ifndef PLUGINBASE_H
#define PLUGINBASE_H

namespace CompuCell3D {

    class PluginBase {
    public:
        virtual ~PluginBase() {}

        virtual void f() = 0;

    };
};
#endif
