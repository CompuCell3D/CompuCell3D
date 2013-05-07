#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <sstream>
#include "rrStringList.h"
#include "rrNewArrayList.h"
#include "rrLogger.h"
//---------------------------------------------------------------------------

namespace rr
{

NewArrayList::NewArrayList()
{}

NewArrayList::NewArrayList(const string& lbl, const StringList& stringList)
{
	Add(lbl, stringList);
}

NewArrayList::NewArrayList(const string& lbl, const NewArrayList& NewArrayList)
{
	Add(lbl, NewArrayList);
}

NewArrayList::~NewArrayList()
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

string NewArrayList::GetString(const int& index)
{
    if(index < mList.size())
    {
		NewArrayListItemObject* listPtr = mList[index];

   		if(listPtr)
        {
        	if(dynamic_cast< NewArrayListItem<string>* >(listPtr))
            {
				return *(dynamic_cast< NewArrayListItem<string>* >(listPtr));
            }
        }
    }

	throw("No string at index");
}

StringList NewArrayList::GetStringList(const int& index)
{
    if(index < mList.size())
    {
		NewArrayListItemObject* listPtr = mList[index];

   		if(listPtr)
        {
        	if(dynamic_cast< NewArrayListItem<StringList>* >(listPtr))
            {
				return *(dynamic_cast< NewArrayListItem<StringList>* >(listPtr));
            }
        }
    }

	throw("No Stringlist at index");
}

StringList NewArrayList::GetStringList(const string& lName)
{
    //Look for ann array list whose first item is a string with lName and second item is a stringlist, i.e. {{string, {string string string}}
    StringList aList;
    for(u_int i = 0; i < Count(); i++)
    {
        NewArrayListItemObject* listPtr = const_cast<NewArrayListItemObject*>(mList[i]);

        //Check for a list which first element is a string, i.e. a {{string}, {string, string}} list
        if(dynamic_cast< NewArrayListItem<NewArrayList> *>(listPtr))
        {
			NewArrayList list = (NewArrayList) *(dynamic_cast< NewArrayListItem<NewArrayList> *>(listPtr));
            if(list.Count())
            {
                NewArrayListItemObject* anItem = &list[0];
                if(dynamic_cast<NewArrayListItem<string>*>(anItem))
                {
                    string str = (string) *dynamic_cast<NewArrayListItem<string>*>(anItem);

                    if(str == lName && list.Count() > 1)
                    {
                        NewArrayListItemObject* anItem = &list[1];
                        if(dynamic_cast<NewArrayListItem<StringList> *>(anItem))
                        {
                            //This is a stringList
                            StringList  list = (StringList) *(dynamic_cast<NewArrayListItem<StringList>*>(anItem));
                            for(int i = 0; i < list.Count(); i++)
                            {
                            	string str = list[i];
                                aList.Add(str);
                            }
                        }
                    }
                }
            }
        }
    }
    return aList;
}

void NewArrayList::Clear()
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

unsigned int NewArrayList::Count() const
{
    return mList.size();
}

NewArrayList::NewArrayList(const NewArrayList& copyMe)
{
    //Deep copy
    Clear();
    mList.resize(copyMe.Count());
    for(u_int i = 0; i < copyMe.Count(); i++)
    {
        //const NewArrayListItem<T>& item = copyMe[i];
        NewArrayListItemObject* ptr = const_cast<NewArrayListItemObject*>(&copyMe[i]);
        if(dynamic_cast<NewArrayListItem<int>*>(ptr))
        {
            mList[i] = new NewArrayListItem<int>(*(dynamic_cast<NewArrayListItem<int>*>(ptr)));
        }
        else if(dynamic_cast<NewArrayListItem<double>*>(ptr))
        {
            mList[i] = new NewArrayListItem<double>(*(dynamic_cast<NewArrayListItem<double>*>(ptr)));
        }
        else if(dynamic_cast<NewArrayListItem<string>*>(ptr))
        {
            mList[i] = new NewArrayListItem<string>(*(dynamic_cast<NewArrayListItem<string>*>(ptr)));
        }
        else if(dynamic_cast<NewArrayListItem<StringList>*>(ptr))
        {
            mList[i] = new NewArrayListItem<StringList>(*(dynamic_cast<NewArrayListItem<StringList>*>(ptr)));
        }
        else if(dynamic_cast<NewArrayListItem<NewArrayList>*>(ptr))
        {
            mList[i] = new NewArrayListItem<NewArrayList>(*(dynamic_cast<NewArrayListItem<NewArrayList>*>(ptr)));
        }

//        else if(dynamic_cast<NewArrayListItem<NewArrayListItem>*>(ptr))
//        {
//            mList[i] = new NewArrayListItem<NewArrayListItem>(*(dynamic_cast<NewArrayListItem<NewArrayListItem>*>(ptr)));
//        }
        else
        {
            mList[i] = NULL;
        }
    }
}

void NewArrayList::operator=(const NewArrayList& rhs)
{
    //Deep copy
    Clear();
    mList.resize(rhs.Count());
    for(u_int i = 0; i < rhs.Count(); i++)
    {
        NewArrayListItemObject* ptr = const_cast<NewArrayListItemObject*>(&rhs[i]);
        if(dynamic_cast<NewArrayListItem<int>*>(ptr))
        {
            mList[i] = new NewArrayListItem<int>(*(dynamic_cast<NewArrayListItem<int>*>(ptr)));
        }
        else if(dynamic_cast<NewArrayListItem<double>*>(ptr))
        {
            mList[i] = new NewArrayListItem<double>(*(dynamic_cast<NewArrayListItem<double>*>(ptr)));
        }
        else if(dynamic_cast<NewArrayListItem<string>*>(ptr))
        {
            mList[i] = new NewArrayListItem<string>(*(dynamic_cast<NewArrayListItem<string>*>(ptr)));
        }
        else if(dynamic_cast<NewArrayListItem<NewArrayList>*>(ptr))
        {
            mList[i] = new NewArrayListItem<NewArrayList>(*(dynamic_cast<NewArrayListItem<NewArrayList>*>(ptr)));
        }

//        else if(dynamic_cast<NewArrayListItem<NewArrayListItem>*>(ptr))
//        {
//            mList[i] = new NewArrayListItem<NewArrayListItem>(*(dynamic_cast<NewArrayListItem<NewArrayListItem>*>(ptr)));
//        }
        else
        {
            mList[i] = NULL;
        }
    }
}

void NewArrayList::Add(const int& item)
{
    NewArrayListItem<int>* ptr =  new NewArrayListItem<int>(item);
    mList.push_back(ptr);
}

void NewArrayList::Add(const double& item)
{
    NewArrayListItem<double>* ptr = new NewArrayListItem<double>(item);
    mList.push_back(ptr);
}

void NewArrayList::Add(const string& item)
{
    NewArrayListItem<string> *ptr = new NewArrayListItem<string>(item);
    mList.push_back(ptr);
}

void NewArrayList::Add(const StringList& item)
{
    NewArrayListItem< StringList > *ptr = new NewArrayListItem<StringList>(item);
    mList.push_back(ptr);
}

void NewArrayList::Add(const NewArrayList& item)
{
    NewArrayListItem<NewArrayList> *aList = new NewArrayListItem<NewArrayList>(item);
    mList.push_back(aList);
}

void NewArrayList::Add(const string& lbl, const StringList& list)
{
    NewArrayList temp;
    temp.Add(lbl);
    temp.Add(list);
    Add(temp);
}

void NewArrayList::Add(const string& lbl, const NewArrayList& lists)
{
    NewArrayList temp;
    temp.Add(lbl);
    temp.Add(lists);
    Add(temp);
}

const NewArrayListItemObject& NewArrayList::operator[](int pos) const
{
    return *mList[pos];
}

NewArrayListItemObject& NewArrayList::operator[](int pos)
{
    return *mList[pos];
}

string NewArrayList::AsString()
{
	stringstream aStr;
    aStr << *this;
    return aStr.str();
}

//================== ostreams =============
ostream& operator<<(ostream& stream, const NewArrayList& list)
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




