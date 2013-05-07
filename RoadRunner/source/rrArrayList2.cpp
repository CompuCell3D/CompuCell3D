#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <sstream>
#include "rrArrayList.h"
#include "rrArrayList2.h"
#include "rrLogger.h"
//---------------------------------------------------------------------------

namespace rr
{

// ============= ArrayList2 ==========================
ArrayList2::ArrayList2()
{}


ArrayList2::~ArrayList2()
{
    if(mList.size())
    {
        for(u_int i = 0; i < Count(); i++)
        {
            delete mList[i];
        }
        mList.clear();
    }
}

void ArrayList2::Add(const string& lbl, const StringList& list)
{
    ArrayList2 temp;
    temp.Add(lbl);
    temp.Add(list);
    Add(temp);
}

void ArrayList2::Add(const StringList& list)
{
    ArrayList2 temp;
    for(int i = 0; i < list.Count(); i++)
    {
        temp.Add(list[i]);
    }
    Add(temp);
}

void ArrayList2::Add(const string& lbl, const StringArrayList& lists)
{

}

StringList ArrayList2::GetSubList(const string& lName)
{
    //Look for list whose first item is lName
    StringList aList;
    for(u_int i = 0; i < Count(); i++)
    {
        ArrayListItemBase* listPtr = const_cast<ArrayListItemBase*>(mList[i]);

        //Check for a list of list
        if(dynamic_cast< ArrayListItem<ArrayList2Item> *>(listPtr))
        {
            ArrayList2Item  list = (ArrayList2Item) *(dynamic_cast<ArrayListItem<ArrayList2Item>*>(listPtr));
            if(list.Count())
            {
                ArrayListItemBase* anItem = &list[0];
                if(dynamic_cast<ArrayListItem<string>*>(anItem))
                {
                    string str = (string) *dynamic_cast<ArrayListItem<string>*>(anItem);

                    if(str == lName && list.Count() > 1)
                    {
                        ArrayListItemBase* anItem = &list[1];
                        if(dynamic_cast< ArrayListItem<ArrayList2Item> *>(anItem))
                        {
                            //This is the sublist of strings..
                            ArrayList2Item  list = (ArrayList2Item) *(dynamic_cast<ArrayListItem<ArrayList2Item>*>(anItem));
                            for(int i = 0; i < list.Count(); i++)
                            {
                                ArrayListItemBase* anItem = &list[i];
                                if(dynamic_cast<ArrayListItem<string>*>(anItem))
                                {
                                    string str = (string) *dynamic_cast<ArrayListItem<string>*>(anItem);
                                    aList.Add(str);
                                }
                            }
                        }
                    }
                }
            }
        }

    }
    return aList;
}

void ArrayList2::Clear()
{
    if(Count())
    {
        for(u_int i = 0; i < Count(); i++)
        {
            delete mList[i];
        }
        mList.clear();
    }
}

unsigned int ArrayList2::Count() const
{
    return mList.size();
}

ArrayList2::ArrayList2(const ArrayList2& copyMe)
{
    //Deep copy
    Clear();
    mList.resize(copyMe.Count());
    for(u_int i = 0; i < copyMe.Count(); i++)
    {
        //const ArrayList2Item<T>& item = copyMe[i];
        ArrayListItemBase* ptr = const_cast<ArrayListItemBase*>(&copyMe[i]);
        if(dynamic_cast<ArrayListItem<int>*>(ptr))
        {
            mList[i] = new ArrayListItem<int>(*(dynamic_cast<ArrayListItem<int>*>(ptr)));
        }
        else if(dynamic_cast<ArrayListItem<double>*>(ptr))
        {
            mList[i] = new ArrayListItem<double>(*(dynamic_cast<ArrayListItem<double>*>(ptr)));
        }
        else if(dynamic_cast<ArrayListItem<string>*>(ptr))
        {
            mList[i] = new ArrayListItem<string>(*(dynamic_cast<ArrayListItem<string>*>(ptr)));
        }
        else if(dynamic_cast<ArrayListItem<ArrayList2Item>*>(ptr))
        {
            mList[i] = new ArrayListItem<ArrayList2Item>(*(dynamic_cast<ArrayListItem<ArrayList2Item>*>(ptr)));
        }
        else
        {
            mList[i] = NULL;
        }
    }
}

void ArrayList2::operator=(const ArrayList2& rhs)
{
    //Deep copy
    Clear();
    mList.resize(rhs.Count());
    for(u_int i = 0; i < rhs.Count(); i++)
    {
        ArrayListItemBase* ptr = const_cast<ArrayListItemBase*>(&rhs[i]);
        if(dynamic_cast<ArrayListItem<int>*>(ptr))
        {
            mList[i] = new ArrayListItem<int>(*(dynamic_cast<ArrayListItem<int>*>(ptr)));
        }
        else if(dynamic_cast<ArrayListItem<double>*>(ptr))
        {
            mList[i] = new ArrayListItem<double>(*(dynamic_cast<ArrayListItem<double>*>(ptr)));
        }
        else if(dynamic_cast<ArrayListItem<string>*>(ptr))
        {
            mList[i] = new ArrayListItem<string>(*(dynamic_cast<ArrayListItem<string>*>(ptr)));
        }
        else if(dynamic_cast<ArrayListItem<ArrayList2Item>*>(ptr))
        {
            mList[i] = new ArrayListItem<ArrayList2Item>(*(dynamic_cast<ArrayListItem<ArrayList2Item>*>(ptr)));
        }
        else
        {
            mList[i] = NULL;
        }
    }
}

void ArrayList2::Add(const int& item)
{
    ArrayListItem<int>* ptr =  new ArrayListItem<int>(item);
    mList.push_back(ptr);
}

void ArrayList2::Add(const double& item)
{
    ArrayListItem<double>* ptr = new ArrayListItem<double>(item);
    mList.push_back(ptr);
}

void ArrayList2::Add(const string& item)
{
    ArrayListItem<string> *ptr = new ArrayListItem<string>(item);
    mList.push_back(ptr);
}

void ArrayList2::Add(const ArrayListItem<ArrayList2Item>& item)
{
    ArrayListItem<ArrayList2Item> *ptr = new ArrayListItem<ArrayList2Item>(item);
    mList.push_back(ptr);
}

void ArrayList2::Add(const ArrayList2& item)
{
    ArrayListItem<ArrayList2Item> *aList = new ArrayListItem<ArrayList2Item>(item);
    mList.push_back(aList);
}

const ArrayListItemBase& ArrayList2::operator[](int pos) const
{
    return *mList[pos];
}

ArrayListItemBase& ArrayList2::operator[](int pos)
{
    return *mList[pos];
}

//================== ostreams =============
ostream& operator<<(ostream& stream, const ArrayList2& list)
{
   	stream<<"{";

    for(u_int i = 0; i < list.Count(); i++)
    {
        stream<<list[i];
        if(i < list.Count() -1)
        {
        	stream<<",";
        }
    }
    stream<<"}";
    return stream;
}

}

