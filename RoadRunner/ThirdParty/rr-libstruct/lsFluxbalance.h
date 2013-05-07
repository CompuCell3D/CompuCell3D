/*! \file   fluxbalance.h
\brief  basic functions for steady state flux balance analysis of SBML files

\par
This file states flux balance analysis of SBML models (or Stoichiometry matrices)
along with flux constraints as linear programming question, which is then solved
using lpsolve (http://sf.net/projects/lpsolve).

\author  Frank T. Bergmann (fbergman@u.washington.edu)

*/

#ifndef FLUX_BALANCE_H
#define FLUX_BALANCE_H

#define FB_EXPORT_LP 1
#define FB_EXPORT_MPS 2
#define FB_EXPORT_FREE_MPS 3

#define FB_OPERATION_LEQ 1
#define FB_OPERATION_GEQ 2
#define FB_OPERATION_EQ 3


#ifdef __cplusplus

#include <string>
#include <vector>
#include <iosfwd>

#include "constraint.h"
#include "objective.h"
#include "lpresult.h"

#include "matrix.h"
#include "libstructural.h"
#include "libutil.h"

namespace ls
{

    /*! \enum ls::ExportFormats 
        \brief enum collecting all exports format that will be written by ls::FluxBalance::writeToFile

        \par 
        After stating the linear programming question, ls::FluxBalance 
        allows to export the question as several standard formats: LP, MPS or FreeMPS

    */
    enum ExportFormats
    {
        LP       = FB_EXPORT_LP, 
        MPS      = FB_EXPORT_MPS,
        FREE_MPS = FB_EXPORT_FREE_MPS

    } FbExport;
    
    /*! \enum ls::ConstraintOperation 
     \brief enum providing human readable interpretions for ls::FluxBalance::addConstraint
     
     \par 
     When specifying the Flux Constraints one is free to choose between three operations: 
     
     \li LessOrEqual, 
     \li GreaterOrEqual, 
     \li Equal
     */
    enum ConstraintOperation {
        LessOrEqual = FB_OPERATION_LEQ, 
        GreaterOrEqual = FB_OPERATION_GEQ, 
        Equal = FB_OPERATION_EQ
    } FbOperation;


    /*! \class ls::FluxBalance
        \brief basic functions for steady state flux balance analysis of SBML files

        \par 
        ls::FluxBalance states flux balance analysis of SBML models (or Stoichiometry matrices)
        along with flux constraints as linear programming question, which is then solved
        using lpsolve (http://sf.net/projects/lpsolve).

    */
    class FluxBalance
    {
    public:
        typedef ls::Matrix< double > DoubleMatrix;

        //! static method to get an instance of LibStructural (allows use as singleton)
        LIB_EXTERN static FluxBalance* getInstance();

        //! construct a new FluxBalance
        LIB_EXTERN FluxBalance(void);
        //! construct a new FluxBalance with given SBML model
        LIB_EXTERN FluxBalance(std::string &sbmlContent);
        //! virtual destructor
        LIB_EXTERN virtual ~FluxBalance(void);

        //! initialize this class with a sbml model passed as string
        LIB_EXTERN void loadSBML(std::string &sbmlContent);

        //! initialize this class with a sbml model loaded from file
        LIB_EXTERN void loadSBMLFromFile(std::string &fileName);

        
        //! initialize class with the given stoichiometry matrix and vector of fluxNames. 
        LIB_EXTERN void loadStoichiometry(ls::DoubleMatrix &matrix, std::vector<std::string> &fluxNames);        

        //! transforms the current model along with constraint and objectives as lp problem and solves the maximization objective using lpsolve.
        LIB_EXTERN LPResult *solve();
        //! transforms the current model along with constraint and objectives as lp problem and solves the minimization objective using lpsolve.
        LIB_EXTERN LPResult *minimize();
        //! transforms the current model along with constraint and objectives as lp problem and solves the maximization objective using lpsolve.
        LIB_EXTERN LPResult *maximize();

        //! deletes all current constraints
        LIB_EXTERN void clearConstraints();
        /*! Adds a new constraint with given fluxName, operation type and constraint value. 
         * \remarks operation is one of 1: LessOrEqual, 2: GreaterOrEqual, 3:Equal
         *          for convenience the enum ls::ConstraintOperation is provided.
         */
        LIB_EXTERN void addConstraint(std::string &id, int operation, double value);
        //! replaces the current constraints with the ones specified in the given constraint vector
        LIB_EXTERN void setConstraints(std::vector< Constraint > constraint);

        //! deletes all current objectives
        LIB_EXTERN void clearObjectives();
        //! adds a new objective with given fluxNames and objective value
        LIB_EXTERN void addObjective(std::string &id, double value);
        //! replaces all current objectives with the ones specified in the given objective vector
        LIB_EXTERN void setObjectives(std::vector< Objective > objectives);

        //! writes the LP problem specified by model, objectives and constraints as file with the given format
        LIB_EXTERN void writeToFile(std::string &fileName, ExportFormats format);

        std::vector< Objective   >    Objectives;
        std::vector< Constraint  >    Constraints;
        std::vector< std::string >    ReactionNames;

        DoubleMatrix *                Stoichiometry;

        char*                        OutputFilename;

    private:
        int getIndex(std::string &id);
        void loadSBMLContentIntoStructAnalysis(std::string &sbmlContent);
        double* SetupLPsolve(void *lp, int numRows, int numColumns);
        static FluxBalance* _Instance;
    };
}

#endif

BEGIN_C_DECLS;

//! initialize this class with a sbml model passed as string
LIB_EXTERN int FluxBalance_loadSBML(const char* sbmlContent);
//! initialize class with the given stoichiometry matrix and vector of fluxNames. 
LIB_EXTERN int FluxBalance_loadStoichiometry(const double** matrix, const int numRows, const int numCols, const char** fluxNames);

//! deletes all current constraints
LIB_EXTERN int FluxBalance_clearConstraints();
/*! Adds a new constraint with given fluxName, operation type and constraint value. 
 * \remarks operation is one of 1: LessOrEqual, 2: GreaterOrEqual, 3:Equal
 *          for convenience the defines FB_OPERATION_LEQ, FB_OPERATION_GEQ and FB_OPERATION_EQ are defined.
 */
LIB_EXTERN int FluxBalance_addConstraint(const char* id, const int operation, const double value);
//! replaces the current constraints with the ones specified in the given constraint vector
LIB_EXTERN int FluxBalance_setConstraints(const char** ids, const int* operations, const double* values, const int length);

//! deletes all current objectives
LIB_EXTERN int FluxBalance_clearObjectives();
//! adds a new objective with given fluxNames and objective value
LIB_EXTERN int FluxBalance_addObjective(const char* id, const double value);
//! replaces all current objectives with the ones specified in the given objective vector
LIB_EXTERN int FluxBalance_setObjectives(const char** ids, const double* values, const int length);

//! transforms the current model along with constraint and objectives as lp problem and solves the maximization objective using lpsolve.
LIB_EXTERN int FluxBalance_solve(char* **outNames, double* *outValues, int *outLength);
//! transforms the current model along with constraint and objectives as lp problem and solves the maximization objective using lpsolve.
LIB_EXTERN int FluxBalance_maximize(char* **outNames, double* *outValues, int *outLength);
//! transforms the current model along with constraint and objectives as lp problem and solves the minimization objective using lpsolve.
LIB_EXTERN int FluxBalance_minimize(char* **outNames, double* *outValues, int *outLength);

//! writes the LP problem specified by model, objectives and constraints as lp file 
LIB_EXTERN void FluxBalance_writeToLPFile(const char* fileName);
//! writes the LP problem specified by model, objectives and constraints as mps file 
LIB_EXTERN void FluxBalance_writeToMPSFile(const char* fileName);
//! writes the LP problem specified by model, objectives and constraints as free mps file 
LIB_EXTERN void FluxBalance_writeToFreeMPSFile(const char* fileName);

END_C_DECLS;


#endif
