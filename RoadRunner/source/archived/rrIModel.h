#ifndef rrIModelH
#define rrIModelH
#include <string>
#include <vector>
#include <list>
#include "rrObject.h"
#include "rrTEventDelayDelegate.h"
#include "rrTEventAssignmentDelegate.h"
#include "rrTComputeEventAssignmentDelegate.h"
#include "rrTPerformEventAssignmentDelegate.h"

using std::list;
using std::vector;
using std::string;

namespace rr
{

class RR_DECLSPEC IModel : public rrObject    //Abstract class for SBML Models to be simulated
{
    protected:
        //These variables is also generated in the c-code, weird ??
        //Init a decendent models data later
        int                                     mDummyInt;
        int                                    *numIndependentVariables;
        int                                    *numDependentVariables;
        int                                    *numTotalVariables;
        int                                    *numBoundaryVariables;
        int                                    *numGlobalParameters;
        int                                    *numCompartments;
        int                                    *numReactions;
        int                                    *numRules;
        int                                    *numEvents;
        string                                  mModelName;


    public:
        double                                  *time;
        void                                    SetTime(double _time){*time = _time;}
        double                                  GetTime(){return *time;}
        vector<double>                          y;
        list<string>                            Warnings;
        vector<double>                          init_y;
        double*                                 m_dydt;               //This is the "dydt" data in the DLL. IModel also has amounts.. CONFUSING
        double*                                 mAmounts;        //This is the "amounts" data in the DLL. IModel also has amounts.. CONFUSING
        virtual double                          GetAmounts(const int& i) = 0;
        double*                                 bc;
        vector<double>                          sr;
        vector<double>                          gp;                //Global parameters
        vector<double>                          lp ;            //Local parameters
        double*                                 c;                //Compartment volumes
        vector<double>                          dydt;
        vector<double>                          rates;
        vector<double>                          ct ;             //Conservation totals
        vector<double>                          rateRules;        //additional rateRules
        vector<double>                          eventTests;
        vector<double>                          eventPriorities;
        vector<TEventDelayDelegate>             eventDelay;
        vector<bool>                            eventType;
        vector<bool>                            eventPersistentType;
        vector<bool>                            eventStatusArray;
        vector<bool>                            previousEventStatusArray;
        vector<TEventAssignmentDelegate>        eventAssignments;
        vector<TComputeEventAssignmentDelegate> computeEventAssignments;
        vector<TPerformEventAssignmentDelegate> performEventAssignments;

                                                IModel();
        virtual                                ~IModel();
        string                                  getModelName();

        // Virtual functions --------------------------------------------------------
        virtual int                             getNumIndependentVariables();
        virtual int                             getNumDependentVariables();
        virtual int                             getNumTotalVariables();
        virtual int                             getNumBoundarySpecies();
        virtual int                             getNumGlobalParameters();
        virtual int                             getNumCompartments();
        virtual int                             getNumReactions();
        virtual int                             getNumRules();
        virtual int                             getNumEvents();
        virtual void                            initializeInitialConditions() = 0;
        virtual void                            setInitialConditions();
        virtual void                            setParameterValues();
        virtual void                            setBoundaryConditions();
        virtual void                            InitializeRates();
        virtual void                            AssignRates();
        virtual void                            AssignRates(vector<double>& rates);
        virtual void                            computeConservedTotals();
        virtual void                            computeEventPriorites();
        virtual void                            setConcentration(int index, double value);
        virtual void                            convertToAmounts();
        virtual void                            convertToConcentrations() = 0;
        virtual void                            updateDependentSpeciesValues(vector<double>& _y);
        virtual void                            computeRules(vector<double>& _y);
        virtual void                            computeReactionRates(double time, vector<double>& y);
        virtual void                            computeAllRatesOfChange();
        virtual void                            evalModel(double time, vector<double>& y);
        virtual void                            evalEvents(double time, vector<double>& y) = 0;
        virtual void                            resetEvents();
        virtual void                            evalInitialAssignments();
        virtual void                            testConstraints();
        virtual void                            InitializeRateRuleSymbols();

        //Pure virtuals - force implementation...
        virtual void                            setCompartmentVolumes() = 0;
        virtual vector<double>                  GetCurrentValues() = 0 ;
        virtual double                          getConcentration(int index) = 0;
        virtual int                             getNumLocalParameters(int reactionId) = 0;         // Level 2 support

        virtual vector<double>                  GetdYdT() = 0;
        virtual void                            LoadData() = 0;    //This one copies data from the DLL to vectors and lists in the model..

};
} //namespace rr

//C#
//    public interface IModel
//    {
//        // Property signatures:
//        double[] y { get; set; }
//
//        List<string> Warnings { get; set; }
//
//
//        double[] init_y { get; set; }
//
//
//        double[] amounts { get; set; }
//
//        double[] bc { get; set; }
//
//        /// <summary>
//        /// modifiable species reference values
//        /// </summary>
//        double[] sr { get; set; }
//
//        // Global parameters
//        double[] gp { get; set; }
//
//        // Local parameters
//        double[][] lp { get; set; }
//
//        // Compartment volumes
//        double[] c { get; set; }
//
//        double[] dydt { get; set; }
//
//        double[] rates { get; set; }
//
//        // Conservation totals
//        double[] ct { get; set; }
//        // additional rateRules
//        double[] rateRules { get; set; }
//
//        double time { get; set; }
//
//        double[] eventTests { get; set; }
//
//        double[] eventPriorities { get; set; }
//
//        TEventDelayDelegate[] eventDelay { get; set; }
//
//        bool[] eventType { get; set; }
//        bool[] eventPersistentType { get; set; }
//
//        bool[] eventStatusArray { get; set; }
//
//        bool[] previousEventStatusArray { get; set; }
//
//        TEventAssignmentDelegate[] eventAssignments { get; set; }
//        TComputeEventAssignmentDelegate[] computeEventAssignments { get; set; }
//        TPerformEventAssignmentDelegate[] performEventAssignments { get; set; }
//
//        int getNumIndependentVariables { get; }
//        int getNumDependentVariables { get; }
//        int getNumTotalVariables { get; }
//        int getNumBoundarySpecies { get; }
//        int getNumGlobalParameters { get; }
//        int getNumCompartments { get; }
//        int getNumReactions { get; }
//        int getNumRules { get; }
//        int getNumEvents { get; }
//
//        void setCompartmentVolumes();
//        void initializeInitialConditions();
//        void setInitialConditions();
//        void setParameterValues();
//        void setBoundaryConditions();
//        void InitializeRates();
//        void AssignRates();
//        void AssignRates(double[] rates);
//        double[] GetCurrentValues();
//        void computeConservedTotals();
//        void computeEventPriorites();
//        void setConcentration(int index, double value);
//        double getConcentration(int index);
//        void convertToAmounts();
//        void convertToConcentrations();
//        void updateDependentSpeciesValues(double[] _y);
//        void computeRules(double[] _y);
//        void computeReactionRates(double time, double[] y);
//        void computeAllRatesOfChange();
//        void evalModel(double time, double[] y);
//        void evalEvents(double time, double[] y);
//        void resetEvents();
//
//        void evalInitialAssignments();
//        void testConstraints();
//        void InitializeRateRuleSymbols();
//
//        // Level 2 support
//        int getNumLocalParameters(int reactionId);
//    }

#endif
