#ifndef rrArrayListH
#define rrArrayListH
//#include <vector>
//#include <string>
//#include <list>
//#include "rrObject.h"
//#include "rrStringList.h"
//
//namespace rr
//{
//
//template <class T>
//class RRArrayList;
//
//template <class T>
//class RRArrayListItem : public rrObject
//{
//    public:
//        T                                          *mValue;
//        RRArrayList< T >                           *mLinkedList;
//        								           	RRArrayListItem(const T& primitive);
//                                                    RRArrayListItem(const RRArrayListItem<T>& item);
//        								           	RRArrayListItem(const RRArrayList<T>* item);
//
//                                                   ~RRArrayListItem();
//        T                                           GetValue() const;
//                                                    operator T();
//        bool                                        HasValue() const;
//        bool                                        HasList() const;
//        string                                      AsString(const string& delimiter = ",") const;
//};
//
//template <class T>
//class RRArrayList : public rrObject
//{
//    protected:
//    public:
//        vector< RRArrayListItem<T>* >		   	    mList;	//Contains list items..
//
//    public:
//                                        			RRArrayList();
//                                        			RRArrayList(const RRArrayList<T>& cpyMe);
//                                        		   ~RRArrayList();
//		mutable
//        typename vector< RRArrayListItem<T>* >::const_iterator mIter;
//
//        int                                         Count() const;
//        void                                        Clear();
//		void                                        Add(const T& item);
//        void							            Add(RRArrayList<T>& subList);
//        void                                        Add(const int& list);
//        void                                        Add(const T& lbl, const RRArrayList<T>& lists);
//
//        void                                        Add(const StringList& list);
//        void                                        Add(const string& lbl, const StringList& lists);
//
//
//        RRArrayListItem<T>&                         operator[](const int& index);
//        const RRArrayListItem<T>&                   operator[](const int& index) const;
//        void                                        operator=(const RRArrayList& rhs);
//        string                                      AsString();
//};
//
//typedef RRArrayList<string> StringArrayList;
//
//template<class T>
//RRArrayList<T>::RRArrayList(){}
//
//template<class T>
//RRArrayList<T>::RRArrayList(const RRArrayList& copyMe)
//{
//    //Copy each item in copyMe
//    mList.resize(copyMe.Count());
//    for(int i = 0; i < copyMe.Count(); i++)
//    {
//        const RRArrayListItem<T>& item = copyMe[i];
//        mList[i] = new RRArrayListItem<T>(item);
//    }
//}
//
//template<class T>
//RRArrayList<T>::~RRArrayList()
//{
//    for(int i = 0; i < Count(); i++)
//    {
//        delete mList[i];
//    }
//    mList.clear();
//}
//
//template<class T>
//void RRArrayList<T>::operator=(const RRArrayList& rhs)
//{
//    Clear();
//
//    //Deep copy..
//    mList.resize(rhs.Count());
//    for(int i = 0; i < rhs.Count(); i++)
//    {
//        const RRArrayListItem<T>& copyItem = rhs[i];
//        RRArrayListItem<T> *item = new RRArrayListItem<T>(copyItem);
//        mList[i] = item;
//    }
//}
//
//template<class T>
//void RRArrayList<T>::Clear()
//{
//    for(int i = 0; i < Count(); i++)
//    {
//        delete mList[i];
//    }
//    mList.clear();
//}
//
//template<class T>
//void RRArrayList<T>::Add(const T& _item)
//{
//	RRArrayListItem<T> *item = new RRArrayListItem<T>(_item);
//    if(item)
//    {
//        mList.push_back(item);
//    }
//}
//
//template<class T>
//void RRArrayList<T>::Add(RRArrayList<T>& subList)
//{
//    RRArrayListItem<T>* newSubList = new RRArrayListItem<T>(&subList);
//
//    //Don't use push back
//    mList.resize(mList.size() + 1);
//    mList[mList.size() - 1] = newSubList;
//}
//
//template<class T>
//void RRArrayList<T>::Add(const T& _item, const RRArrayList<T>& subList)
//{
//    RRArrayListItem<T>* newSubList = new RRArrayListItem<T>(&subList);
//
//    //Don't use push back
//    mList.resize(mList.size() + 1);
//    mList[mList.size() - 1] = newSubList;
//}
//
////Adding stringlists...
//template<class T>
//void RRArrayList<T>::Add(const StringList& lists)
//{
//
//}
//
//template<class T>
//void RRArrayList<T>::Add(const string& lbl, const StringList& lists)
//{
//
//}
//
//
//template<class T>
//int RRArrayList<T>::Count() const
//{
//    return mList.size();
//}
//
//template<class T>
//string RRArrayList<T>::AsString()
//{
//    string theList;
//    for(int i = 0; i < Count(); i++)
//    {
//        string item = (*this)[i];
//
//        theList += item;
//        if(i < Count() -1)
//        {
//            theList += ",";
//        }
//    }
//    return theList;
//}
//
//////////////////////////////////////////////////////////////////////////////
//template< class T >
//RRArrayListItem<T>::RRArrayListItem(const RRArrayListItem<T>& item)
//{
//    if(item.HasValue())
//    {
//        mValue = new T(item.GetValue());
//        mLinkedList = NULL;
//    }
//    else if (item.mLinkedList)
//    {
//        mLinkedList = new RRArrayList<T>(*item.mLinkedList);
//        mValue = NULL;
//    }
//    else
//    {
//        mValue = NULL;
//        mLinkedList = NULL;
//    }
//}
//
//template< class T >
//RRArrayListItem<T>::RRArrayListItem(const T& item)
//:
//mLinkedList(NULL)
//{
//    mValue = new T(item);
//}
//
//template< class T >
//RRArrayListItem<T>::~RRArrayListItem()
//{
//    delete mValue;
//    delete mLinkedList;
//}
//
//template< class T >
//RRArrayListItem<T>::RRArrayListItem(const RRArrayList<T>* item)
//:
//mValue(NULL)
//{
//    mLinkedList =  (item) ?  new RRArrayList<T>(*item) : NULL;
//}
//
//template<class T>
//RRArrayListItem<T>& RRArrayList<T>::operator[](const int& index)
//{
//    RRArrayListItem<T> *item = mList[index];
//    return *item;
//}
//
//template<class T>
//const RRArrayListItem<T>&  RRArrayList<T>::operator[](const int& index) const
//{
//    RRArrayListItem<T> *item = mList[index];
//    return *item;
//}
//
//template<class T>
//ostream& operator<<(ostream& stream, RRArrayList<T>& list)
//{
//    int i = 0;
//   	stream<<"{";
//    for(list.mIter = list.mList.begin(); list.mIter != list.mList.end(); list.mIter++)
//    {
//        RRArrayListItem<T>* item = (*list.mIter);
//        if(item->mLinkedList != NULL)
//        {
//            stream<<*item->mLinkedList;
//        }
//
//        if(item->mValue)
//        {
//            stream<< *item->mValue;
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
//template<class T>
//ostream& operator<<(ostream& stream, RRArrayListItem<T>& listItem)
//{
//    if(listItem.mValue)
//    {
//        stream<<listItem.mValue;
//    }
//    return stream;
//}
//
//template <class T>
//T RRArrayListItem<T>::GetValue() const
//{
//    return *mValue;
//}
//
//
//template <class T>
//bool RRArrayListItem<T>::HasValue() const
//{
//    return mValue == NULL ? false : true;
//}
//
//template <class T>
//bool RRArrayListItem<T>::HasList() const
//{
//    return mLinkedList == NULL ? false : true;
//}
//
//}
#endif
