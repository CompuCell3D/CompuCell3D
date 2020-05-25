#include <sstream>

namespace  CompuCell3D
{

    template<typename T>
    std::string to_str(const T& value) {
        std::ostringstream tmp_str;
        tmp_str << value;
        return tmp_str.str();
    }

    template<typename T, typename ... Args >
    std::string to_str(const T& value, const Args& ... args) {
        return to_str(value) + to_str(args...);
    }
}