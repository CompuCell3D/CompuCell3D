#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrUtils.h"
#include "rrModelState.h"
//---------------------------------------------------------------------------

namespace rr
{
ModelState::ModelState(ModelFromC& model)
{
    InitializeFromModel(model);
}

void ModelState::InitializeFromModel(ModelFromC& model)
{
    model.convertToConcentrations();
//    CopyCArrayToStdVector(model.y,                        mFloatingSpeciesConcentrations,       *model.ySize);
    CopyCArrayToStdVector(model.mData.bc,                       mBoundarySpeciesConcentrations,       model.mData.bcSize);
    CopyCArrayToStdVector(model.mData.c,                        mCompartmentVolumes,                  model.mData.cSize);
    CopyCArrayToStdVector(model.mData.gp,                       mGlobalParameters,                    model.mData.gpSize);
    CopyCArrayToStdVector(model.mData.ct,                       mConservedTotals,                     model.mData.ctSize);
    CopyCArrayToStdVector(model.mData.dydt,                     mDyDt,                                model.mData.dydtSize);
    CopyCArrayToStdVector(model.mData.rates,                    mRates,                               model.mData.ratesSize);
    CopyCArrayToStdVector(model.mData.rateRules,                mRateRules,                           model.mData.rateRulesSize);
    CopyCArrayToStdVector(model.mData.sr,                       mModifiableSpeciesReferences,         model.mData.srSize);
    CopyCArrayToStdVector(model.mData.eventStatusArray,         mEventStatusArray,                    model.mData.eventStatusArraySize);
    CopyCArrayToStdVector(model.mData.eventTests,               mEventTests,                          model.mData.eventTestsSize);
    CopyCArrayToStdVector(model.mData.previousEventStatusArray, mPreviousEventStatusArray,            model.mData.previousEventStatusArraySize);
    mTime = model.mData.time;
}

void ModelState::AssignToModel(ModelFromC& model)
{
//    CopyStdVectorToCArray(mFloatingSpeciesConcentrations,   model.y,                        *model.ySize                        );
    CopyStdVectorToCArray(mBoundarySpeciesConcentrations,   model.mData.bc,                       model.mData.bcSize                       );
    CopyStdVectorToCArray(mCompartmentVolumes,              model.mData.c,                        model.mData.cSize                        );
    CopyStdVectorToCArray(mGlobalParameters,                model.mData.gp,                       model.mData.gpSize                       );
    CopyStdVectorToCArray(mConservedTotals,                 model.mData.ct,                       model.mData.ctSize                       );
    CopyStdVectorToCArray(mDyDt,                            model.mData.dydt,                     model.mData.dydtSize                     );
    CopyStdVectorToCArray(mRates,                           model.mData.rates,                    model.mData.ratesSize                    );
    CopyStdVectorToCArray(mRateRules,                       model.mData.rateRules,                model.mData.rateRulesSize                );
    CopyStdVectorToCArray(mEventTests,                      model.mData.eventTests,               model.mData.eventTestsSize               );
    CopyStdVectorToCArray(mEventStatusArray,                model.mData.eventStatusArray,         model.mData.eventStatusArraySize         );
    CopyStdVectorToCArray(mPreviousEventStatusArray,        model.mData.previousEventStatusArray, model.mData.previousEventStatusArraySize );
    CopyStdVectorToCArray(mModifiableSpeciesReferences,     model.mData.sr,                       model.mData.srSize                        );
    model.convertToAmounts();
    model.setTime(mTime);
}

}


