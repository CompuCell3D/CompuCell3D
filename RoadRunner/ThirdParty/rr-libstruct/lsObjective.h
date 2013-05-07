/*! \file   objective.h
\brief  basic storage class for constraints used by ls::FluxBalance

\par
Objectives consist of an Id and value that has to be met. 

\author  Frank T. Bergmann (fbergman@u.washington.edu)

*/
#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <string>
#include "libutil.h"

namespace ls
{

    /*! \class ls::Objective
    \brief basic storage class for constraints used by ls::FluxBalance

    \par 
    ls::Objective consist of an Id and value that has to be met. 


    */
    class Objective
    {
    public:
        //! create a new objective
        LIB_EXTERN Objective(void);
        //! create a new objective with the given values
        LIB_EXTERN Objective(std::string &id, double value) : Id(id), Value(value) {}
        //! virtual destructor
        LIB_EXTERN virtual ~Objective(void);

        //! get the flux name
        LIB_EXTERN std::string getId() { return Id;}
        //! set the flux name for this constraint
        LIB_EXTERN void setId(std::string id) { Id = id; }

        //! return the current objective value
        LIB_EXTERN double getValue() { return Value; } 
        //! set the current objective value
        LIB_EXTERN void setValue(double value) { Value = value; } 

        std::string Id;
        double Value;

    };

}

#endif 
