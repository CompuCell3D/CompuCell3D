#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrMisc.h"

//---------------------------------------------------------------------------
namespace rr
{

TSelectionRecord::TSelectionRecord(const int& _index, const TSelectionType _type, const string& _p1, const string& _p2)
:
index(_index),
p1(_p1),
p2(_p2),
selectionType(_type)
{}

ostream& operator<< (ostream& stream, const TSelectionRecord& rec)
{
    stream<<"A Selection Record --"<<endl;
    stream<<"Index: "<<rec.index<<endl;
    stream<<"p1: "<<rec.p1<<endl;
    stream<<"p2: "<<rec.p1<<endl;
    stream<<"SelectionType: "<<rec.selectionType<<endl;
    return stream;
}

}
