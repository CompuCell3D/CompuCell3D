template<typename C, typename RVT>
//Adapter for STL containter Iterator that allows to specify value type (in STLPyIterator return value is set ot container::value_type &)
//RVT -stands for Return Value Type
class STLPyMapIteratorValueType
{
public:

    typename C::iterator current;
    typename C::iterator begin;
    typename C::iterator end;


    STLPyMapIteratorValueType(C& a)
    {
      initialize(a);
    }

    STLPyMapIteratorValueType()
    {
    }


    RVT  getCurrentRef(){
		 return /*const_cast<RVT>*/(std::make_pair(current->first,current->second));
    }


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


