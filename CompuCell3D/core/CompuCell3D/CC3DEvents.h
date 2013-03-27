#ifndef EVENTS_H
#define EVENTS_H


namespace CompuCell3D {

  enum CC3DEvent_t { BASE, LATTICE_RESIZE}; 
  
  
  class CC3DEvent{
      public:
        CC3DEvent(){
            id=BASE;
        }
        CC3DEvent_t id;
      
   };


  class CC3DEventLatticeResize:public CC3DEvent{
      public:
        CC3DEventLatticeResize(){
            id=LATTICE_RESIZE;
        }
        
        Dim3D newDim;
        Dim3D oldDim;
        Dim3D shiftVec;
      
   };
   
  

};
#endif