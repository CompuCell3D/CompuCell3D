

#include "Field3D.h"

using namespace CompuCell3D;
template<>
const char Field3D<int>::typeStr[4] = " i";
template<>
const char Field3D<unsigned int>::typeStr[4] = "ui";
template<>
const char Field3D<double>::typeStr[4] = " d";
template<>
const char Field3D<float>::typeStr[4] = " f";
template<>
const char Field3D<char>::typeStr[4] = " c";
template<>
const char Field3D<unsigned char>::typeStr[4] = "uc";
template<>
const char Field3D<long>::typeStr[4] = " l";
template<>
const char Field3D<unsigned long>::typeStr[4] = "ul";
