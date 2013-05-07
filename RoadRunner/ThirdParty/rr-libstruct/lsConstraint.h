/*! \file   constraint.h
\brief  basic storage class for constraints used by ls::FluxBalance

\par
Constraints consist of an Id, operation type and value that has to be met. 

\author  Frank T. Bergmann (fbergman@u.washington.edu)

*/
#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include <string>
#include "libutil.h"

namespace ls
{
    /*! \class ls::Constraint
    \brief basic storage class for constraints used by ls::FluxBalance

    \par 
    ls::Constraint consist of an Id, operation type and value that has to be met. 


    */
    class Constraint
    {
    public:
        //! create a new constraint
        LIB_EXTERN Constraint(void);
        //! create a new constraint with the given values
        LIB_EXTERN Constraint(std::string &id, int op, double value) : Id(id), Operator(op), Value(value) {}
        //! virtual destructor
        LIB_EXTERN virtual ~Constraint(void);

        //! get the flux name
        LIB_EXTERN std::string getId() { return Id;}
        //! set the flux name for this constraint
        LIB_EXTERN void setId(std::string id) { Id = id; }

        //! return the current constraint value
        LIB_EXTERN double getValue() { return Value; } 
        //! set the current constraint value
        LIB_EXTERN void setValue(double value) { Value = value; } 

        //! gets the current constraint operation
        //\remarks operation is one of 1: LessOrEqual, 2: GreaterOrEqual, 3:Equal
        LIB_EXTERN int getOperator() { return Operator; } 
        //! set the current constraint operation
        //\remarks operation is one of 1: LessOrEqual, 2: GreaterOrEqual, 3:Equal
        LIB_EXTERN void setOperator(int op) { Operator = op; } 

        std::string Id;
        int Operator;
        double Value;
    };

}

#endif
