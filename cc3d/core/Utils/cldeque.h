#ifndef CLDEQUE_H
#define CLDEQUE_H
#include <deque>
#include <iostream>

template<typename T>
class cldeque{
   public:
   typedef typename std::deque<T>::size_type size_type;
      cldeque()
      {}
      cldeque(size_type _n):
         cld(_n)
      {}
      cldeque(size_type _n,const T & _t):
         cld(_n,_t)
      {}
      void setSize(size_type _n){///it will resize deque
         if(cld.size()>_n){
            while(cld.size()>_n)
               cld.pop_back();
         }
         if(cld.size()<_n){
            while(cld.size()<_n)
               cld.push_back(T());
         }
      }
      void push_front(const T & _t){
            cld.pop_back();
            cld.push_front(_t);

      }
      const T & operator[](size_type _idx) const {return cld[_idx];}
      T & operator[](size_type _idx){return cld[_idx];}
      void assign(size_type _size,const T & _t){
         cld.assign(_size,_t);
      }
      
      size_type size() const {return cld.size();}
   private:
      std::deque<T> cld;
};

template<typename T>
std::ostream & operator<<(std::ostream &_out,cldeque<T> & _cldeque){
    using namespace std;
   for(unsigned int i = 0 ; i < _cldeque.size() ; ++i){
      _out<<_cldeque[i]<<endl;
   }   
}

///For some reason
///cerr<<deque<<endl; produces segfault if one removes endl it is fine
/// will check it later

#endif
