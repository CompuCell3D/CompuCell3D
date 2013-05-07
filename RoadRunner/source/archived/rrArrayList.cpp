#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <sstream>
#include "rrArrayList.h"
//---------------------------------------------------------------------------

//namespace rr
//{
//
//
//template <>
//RRArrayListItem<string>::operator string()
//{
//    if(mValue)
//    {
//        return *mValue;
//    }
//
//    if(mLinkedList)
//    {
//        return mLinkedList->AsString();
//    }
//    return "";
//}
//
//template <>
//RRArrayListItem<int>::operator int()
//{
//    if(mValue)
//    {
//        return *mValue;
//    }
//
//    if(mLinkedList)
//    {
//        return mLinkedList->operator [](0);
//    }
//    return -1;
//}
//
//template<>
//ostream& operator<<(ostream& stream, RRArrayList<string>& list)
//{
//    int i = 0;
//   	stream<<"{";
//    for(list.mIter = list.mList.begin(); list.mIter != list.mList.end(); list.mIter++)
//    {
//        RRArrayListItem<string>* item = (*list.mIter);
//        if(item->mLinkedList != NULL)
//        {
//            stream<<*item->mLinkedList;
//        }
//
//        if(item->mValue)
//        {
//            stream<<"\""<< *item->mValue <<"\""; //Need to quote strings in order to separate them 'visually' on output, i.e. {S1, {CC:S1,k1, CC:S1,k2 becomes {"S1", {"CC:S1,k1", "CC:S1,k2
//        }
//
//        if(i < list.Count() -1)
//        {
//        	stream<<",";
//        }
//        i++;
//    }
//    stream<<"}";
//    return stream;
//}
//
//
//
//}

