#ifndef rrModelDataH
#define rrModelDataH

#if defined __cplusplus
namespace rr
{
#endif
typedef struct   SModelData *ModelDataP;
typedef double 	(*TEventDelayDelegate)(ModelDataP);
typedef double* (*TComputeEventAssignmentDelegate)(ModelDataP);
typedef void 	(*TPerformEventAssignmentDelegate)(ModelDataP, double*);
typedef void 	(*TEventAssignmentDelegate)();

//#pragma pack(push, 1)
//Data that is used in SBML models
typedef struct SModelData
{
    double	                       	    time;
    int                            	    numIndependentVariables;
    int                            	    numDependentVariables;
    int                            	    numTotalVariables;
    int                            	    numBoundaryVariables;
    int                            	    numGlobalParameters;
    int                            	    numCompartments;
    int                            	    numReactions;
    int                            	    numRules;
    int                            	    numEvents;

    char**                              variableTable;
    char**                              boundaryTable;
    char**                              globalParameterTable;

	//These need allocation...
	char*		                        modelName;

    int                                 ySize;
    double*                             y;

 	int                                 gpSize;
	double* 	                        gp;

	int		 	                        srSize;
	double* 	                        sr;

    int									lpSize;
    double*		                        lp;

	int									init_ySize;
    double*		                        init_y;

    int									amountsSize;
    double*	                            amounts;

    int									bcSize;
    double*	                            bc;

    int		                            cSize;
    double*	                            c;

	int                                 dydtSize;
    double*	                            dydt;

    int									ratesSize;
    double*	                            rates;

    int									rateRulesSize;
    double*	                            rateRules;

	int									ctSize;
    double*	                            ct;

    int                           	    localParameterDimensionsSize;
    int*                           	    localParameterDimensions;

	//Event stuff
	int									eventTypeSize;
    bool*                          	    eventType;

	int									eventPersistentTypeSize;
    bool*                          	    eventPersistentType;

	int									eventTestsSize;
    double*	                            eventTests;

	int									eventPrioritiesSize;
	double*	                     	    eventPriorities;

    int		                            eventStatusArraySize;
    bool*                               eventStatusArray;

	int									previousEventStatusArraySize;
    bool*                               previousEventStatusArray;

    TEventDelayDelegate*                eventDelays;
	TEventAssignmentDelegate*       	eventAssignments;

    TComputeEventAssignmentDelegate* 	computeEventAssignments;
    TPerformEventAssignmentDelegate*    performEventAssignments;

} ModelData;
//#pragma pack(pop)

#if defined __cplusplus
}
#endif



#endif
