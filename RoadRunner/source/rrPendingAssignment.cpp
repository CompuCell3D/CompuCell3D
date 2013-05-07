#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
//#include "rrModelData.h"
#include "rrPendingAssignment.h"
//---------------------------------------------------------------------------

namespace rr
{
PendingAssignment::PendingAssignment(
					SModelData* md,
                    double time,
                    TComputeEventAssignmentDelegate computeAssignment,
                    TPerformEventAssignmentDelegate performAssignment,
                    bool useValuesFromTriggerTime,
                    int index)
:
mModelData(md)
{
    Time = time;
    ComputeAssignment = computeAssignment;
    PerformAssignment = performAssignment;
    Index = index;
    UseValuesFromTriggerTime = useValuesFromTriggerTime;
    if (useValuesFromTriggerTime)
    {
        ComputedValues = computeAssignment(mModelData);
    }
}



void PendingAssignment::AssignToModel()
{
    if (!UseValuesFromTriggerTime)
    {
        ComputedValues = ComputeAssignment(mModelData);
    }
    PerformAssignment(mModelData, ComputedValues);
}

int PendingAssignment::GetIndex()
{
    return Index;
}

double PendingAssignment::GetTime()
{
    return Time;
}

}
