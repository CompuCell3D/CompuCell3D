template<typename C, typename RVT>
class STLPyIteratorMap
{
public:

    typename C::iterator current;
    typename C::iterator begin;
    typename C::iterator end;


    STLPyIteratorMap(C& a)
    {
      initialize(a);
    }

    STLPyIteratorMap()
    {
    }


    RVT getCurrentRef(){
      // return const_cast<typename RVT>(current->second);
      return (RVT)(current->second);
    }
//     typename C::value_type getCurrentInstance(){
//       return const_cast<typename C::value_type>(*current);
//     }

    void initialize(C& a){
        begin = a.begin();
        end = a.end();
    }
    bool isEnd(){return current==end;}
    bool isBegin(){return current==begin;}
    void setToBegin(){current=begin;}
    typename C::iterator & getCurrent(){return current;}
    typename C::iterator & getEnd(){return end;}
//  previous function is never used and we cannot decrement iterators for e.g. unordered_map so commenting this out
//    void previous(){
//        if(current != begin){
//
//            --current;
//         }
//
//    }
    void next()
    {
        
        if(current != end){
        
            ++current;
         }


    }
};


