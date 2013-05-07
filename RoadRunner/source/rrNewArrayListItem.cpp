#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrNewArrayListItem.h"
#include "rrNewArrayList.h"
//---------------------------------------------------------------------------

namespace rr
{

ostream& operator<<(ostream& stream, const NewArrayListItemObject& item)
{
    //Have to figure out subtype of item
    NewArrayListItemObject* ptr = const_cast<NewArrayListItemObject*>(&item);
    if(dynamic_cast<NewArrayListItem<int>*>(ptr))
    {
        stream << (int) *(dynamic_cast<NewArrayListItem<int>*>(ptr));
    }
    else if(dynamic_cast<NewArrayListItem<double>*>(ptr))
    {
        stream << (double) *(dynamic_cast<NewArrayListItem<double>*>(ptr));
    }
    else if(dynamic_cast<NewArrayListItem<string>*>(ptr))
    {
        stream << "\""<<(string) *(dynamic_cast<NewArrayListItem<string>*>(ptr))<<"\"";
    }
    else if(dynamic_cast<NewArrayListItem<StringList>*>(ptr))
    {
        stream << (StringList) *(dynamic_cast<NewArrayListItem<StringList>*>(ptr));
    }
    else if(dynamic_cast<NewArrayListItem<NewArrayList>*>(ptr))
    {
        stream << (NewArrayList) *(dynamic_cast<NewArrayListItem<NewArrayList>*>(ptr));
    }
    else
    {
        stream<<"Stream operator not implemented for this type";
    }
    return stream;
}

}





