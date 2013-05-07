#ifndef rrTParameterTypeH
#define rrTParameterTypeH

namespace rr
{
    enum TParameterType
    {
        ptGlobalParameter = 0,
        ptLocalParameter,
        ptBoundaryParameter,
        ptConservationParameter,
        ptFloatingSpecies
    };
}

#endif

//c#
//namespace LibRoadRunner
//{
//    internal enum TParameterType
//    {
//        ptGlobalParameter,
//        ptLocalParameter,
//        ptBoundaryParameter,
//        ptConservationParameter,
//        ptFloatingSpecies
//    } ;
//}

