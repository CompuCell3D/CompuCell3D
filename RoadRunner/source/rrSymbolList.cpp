#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrSymbolList.h"
//---------------------------------------------------------------------------



namespace rr
{
void SymbolList::Clear()
{
    clear();
}

int SymbolList::Add(const Symbol& item)
{
    push_back(item);
    return size();
}

double SymbolList::getValue(const int& index)
{
    return at(index).value;
}

string SymbolList::getName(const int& index)
{
//            return ((Symbol) base[index]).name;
    return at(index).name;
}

string SymbolList::getKeyName(const int& index)
{
//            return ((Symbol) base[index]).keyName;
    return at(index).keyName;
}


bool SymbolList::find(const string& name, int& index)
{
    index = -1;
    for (unsigned int i = 0; i < size(); i++)
    {
        Symbol sym = at(i);
        if (name == sym.name)
        {
            index = i;
            return true;
        }
    }
    return false;
}

bool SymbolList::find(const string& keyName, const string& name, int& index)
{
    index = -1;
    for(unsigned int i = 0; i < size(); i++)
    {
        Symbol sym = at(i);
        if ((sym.name == name) && (sym.keyName == keyName))
        {
            index = i;
            return true;
        }
    }
    return false;
}

}//namespace rr



