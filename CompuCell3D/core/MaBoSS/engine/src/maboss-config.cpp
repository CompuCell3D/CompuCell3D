#include "maboss-config.h"

long sysconf(int _name) {
    if (_name == _SC_CLK_TCK) return CLOCKS_PER_SEC;
    else {
        std::cerr << "Unknown sysconf name : " << _name << std::endl;
        throw 0;
    }
    return 0;
}
