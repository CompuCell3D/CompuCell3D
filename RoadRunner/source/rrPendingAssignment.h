#ifndef rrPendingAssignmentH
#define rrPendingAssignmentH
#include <vector>
#include "rrObject.h"
#include "rrModelData.h"
//#include "rrTComputeEventAssignmentDelegate.h"
//#include "rrTPerformEventAssignmentDelegate.h"

using std::vector;

//---------------------------------------------------------------------------
// <summary>
// Initializes a new instance of the PendingAssignment class.
// </summary>
// <param name="time"></param>
namespace rr
{

class RR_DECLSPEC PendingAssignment : public rrObject
{
    protected:
      	SModelData*							mModelData;
        double                              Time;
        int                                 Index;
        bool                                UseValuesFromTriggerTime;
        TComputeEventAssignmentDelegate     ComputeAssignment;
        TPerformEventAssignmentDelegate     PerformAssignment;
        int                                 ComputedValuesSize;

    public:
        double*                             ComputedValues;

          /// <summary>
        /// Initializes a new instance of the PendingAssignment class.
        /// </summary>
        /// <param name="time"></param>
                                            PendingAssignment(  SModelData* md, double time,
                                                                TComputeEventAssignmentDelegate computeAssignment,
                                                                TPerformEventAssignmentDelegate performAssignment,
                                                                bool useValuesFromTriggerTime,
                                                                int index);
        int                                 GetIndex();
        double                              GetTime();
        void                                AssignToModel();
};

}
#endif

//namespace LibRoadRunner
//{
//    internal class PendingAssignment
//    {
//        /// <summary>
//        /// Initializes a new instance of the PendingAssignment class.
//        /// </summary>
//        /// <param name="time"></param>
//        public PendingAssignment(double time, TComputeEventAssignmentDelegate computeAssignment,
//                                 TPerformEventAssignmentDelegate performAssignment, bool useValuesFromTriggerTime, int index)
//        {
//            Time = time;
//            ComputeAssignment = computeAssignment;
//            PerformAssignment = performAssignment;
//            Index = index;
//            UseValuesFromTriggerTime = useValuesFromTriggerTime;
//            if (useValuesFromTriggerTime)
//                ComputedValues = computeAssignment();
//        }
//
//        public int Index { get; set; }
//
//        public double Time { get; set; }
//
//        public double[] ComputedValues { get; set; }
//        public TComputeEventAssignmentDelegate ComputeAssignment { get; set; }
//        public TPerformEventAssignmentDelegate PerformAssignment { get; set; }
//        public bool UseValuesFromTriggerTime { get; set; }
//
//        public void AssignToModel()
//        {
//            if (!UseValuesFromTriggerTime)
//                ComputedValues = ComputeAssignment();
//            PerformAssignment(ComputedValues);
//        }
//    }
//}

