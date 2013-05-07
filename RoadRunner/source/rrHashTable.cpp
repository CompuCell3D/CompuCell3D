#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrHashTable.h"
//---------------------------------------------------------------------------

namespace rr
{

StringSymbolHashTable::StringSymbolHashTable()
{}


bool StringSymbolHashTable::ContainsKey(const string& aKey)
{
    return (this->find( aKey ) != this->end()) ? true : false;
}


/////////////////////
IntStringHashTable::IntStringHashTable()
{}

ostream& operator<<(ostream& stream, StringSymbolHashTable& hash)
{
    map<string, SBMLSymbol>::iterator iter;

    for(iter = hash.begin(); iter != hash.end(); iter++)
    {
        stream<<"Key: "<<(*iter).first<<"\tValue:"<<(*iter).second<<"\n";
    }
    return stream;
}



}
