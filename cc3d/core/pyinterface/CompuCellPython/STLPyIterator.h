template<typename C>
class STLPyIterator
{
public:

    typename C::iterator current;
    typename C::iterator begin;
    typename C::iterator end;


    STLPyIterator(C& a)
    {
      initialize(a);
    }

    STLPyIterator()
    {
    }


    typename C::value_type& getCurrentRef(){
      return const_cast<typename C::value_type&>(*current);
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
    void previous(){
        if(current != begin){
        
            --current;
         }
      
    }
    void next()
    {
        
        if(current != end){
        
            ++current;
         }


    }
};


