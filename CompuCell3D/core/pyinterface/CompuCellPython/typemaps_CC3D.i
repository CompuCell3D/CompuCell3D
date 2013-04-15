// typemap(in) CompuCell3D::Dim3D will not overshadow earlier default conversion of list to std::vector
%typemap(in) CompuCell3D::Dim3D (CompuCell3D::Dim3D dim)  {
  /* Check if is a list */
    if (PyList_Check($input)) {
        int size = PyList_Size($input);        
        if (size==3){
            
            dim.x= (short)PyInt_AsLong(PyList_GetItem($input,0));
            dim.y=(short)PyInt_AsLong(PyList_GetItem($input,1));
            dim.z=(short)PyInt_AsLong(PyList_GetItem($input,2));
            $1=dim;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case

        }

    }else if (PyTuple_Check($input)){
        //check if it is a tuple
        int size = PyTuple_Size($input);        
        if (size==3){
            
            dim.x= (short)PyInt_AsLong(PyTuple_GetItem($input,0));
            dim.y=(short)PyInt_AsLong(PyTuple_GetItem($input,1));
            dim.z=(short)PyInt_AsLong(PyTuple_GetItem($input,2));
            $1=dim;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
        }                
    }else{
        
         int res = SWIG_ConvertPtr($input,(void **) &$1, $&1_descriptor,0);
         
         
        if (SWIG_IsOK(res)) {
            
            dim.x=(short)PyInt_AsLong(PyObject_GetAttrString($input,"x"));
            dim.y=(short)PyInt_AsLong(PyObject_GetAttrString($input,"y"));
            dim.z=(short)PyInt_AsLong(PyObject_GetAttrString($input,"z"));
            $1=dim;
        } else {
        
            SWIG_exception(SWIG_ValueError,"Expected CompuCell.Dim3D object."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
            
          
        }
         
    }
}

%typemap(in) CompuCell3D::Dim3D &  (CompuCell3D::Dim3D dim)  { // note that (CompuCell3D::Point3D pt) causes pt to be allocated on the stack - no need to worry abuot freeing memory
  /* Check if is a list */
  // cerr<<"inside Dim3D conversion typemap"<<endl;
    if (PyList_Check($input)) {
        // cerr<<"GOT LIST"<<endl;
        int size = PyList_Size($input);        
        if (size==3){
            
            dim.x= (short)PyInt_AsLong(PyList_GetItem($input,0));
            dim.y=(short)PyInt_AsLong(PyList_GetItem($input,1));
            dim.z=(short)PyInt_AsLong(PyList_GetItem($input,2));
            $1=&dim;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case

        }

    }else if (PyTuple_Check($input)){
        //check if it is a tuple
        int size = PyTuple_Size($input);        
        // cerr<<"GOT TUPLE"<<endl;
        if (size==3){            
            dim.x= (short)PyInt_AsLong(PyTuple_GetItem($input,0));
            dim.y=(short)PyInt_AsLong(PyTuple_GetItem($input,1));
            dim.z=(short)PyInt_AsLong(PyTuple_GetItem($input,2));
            $1=&dim;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
        }                
    }else{
        // cerr<<" CHECKING FOR DIM3D"<<endl;
         int res = SWIG_ConvertPtr($input,(void **) &$1, $1_descriptor,0);
         
         
        if (SWIG_IsOK(res)) {            
            dim.x=(short)PyInt_AsLong(PyObject_GetAttrString($input,"x"));
            dim.y=(short)PyInt_AsLong(PyObject_GetAttrString($input,"y"));
            dim.z=(short)PyInt_AsLong(PyObject_GetAttrString($input,"z"));
            $1=&dim;
        } else {
        
            SWIG_exception(SWIG_ValueError,"Expected CompuCell.Dim3D object."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
                      
        }
         
    }
}

// we dont really need this
// %typemap(in) CompuCell3D::Dim3D *  (CompuCell3D::Dim3D dim) = CompuCell3D::Dim3D &  (CompuCell3D::Dim3D dim); 



%typemap(in) CompuCell3D::Point3D  (CompuCell3D::Point3D pt)  {
  /* Check if is a list */
  // cerr<<"inside point3D conversion typemap"<<endl;
    if (PyList_Check($input)) {
        int size = PyList_Size($input);        
        if (size==3){
            // CompuCell3D::Point3D pt;    
            pt.x= (short)PyInt_AsLong(PyList_GetItem($input,0));
            pt.y=(short)PyInt_AsLong(PyList_GetItem($input,1));
            pt.z=(short)PyInt_AsLong(PyList_GetItem($input,2));
            $1=pt;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case

        }

    }else if (PyTuple_Check($input)){
        //check if it is a tuple
        int size = PyTuple_Size($input);        
        if (size==3){
            // CompuCell3D::Point3D pt;    
            pt.x= (short)PyInt_AsLong(PyTuple_GetItem($input,0));
            pt.y=(short)PyInt_AsLong(PyTuple_GetItem($input,1));
            pt.z=(short)PyInt_AsLong(PyTuple_GetItem($input,2));
            $1=pt;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
        }                
    }else{
        
         int res = SWIG_ConvertPtr($input,(void **) &$1, $&1_descriptor,0);
         
         
        if (SWIG_IsOK(res)) {
            // CompuCell3D::Point3D pt;    
            pt.x=(short)PyInt_AsLong(PyObject_GetAttrString($input,"x"));
            pt.y=(short)PyInt_AsLong(PyObject_GetAttrString($input,"y"));
            pt.z=(short)PyInt_AsLong(PyObject_GetAttrString($input,"z"));
            $1=pt;
        } else {
        
            SWIG_exception(SWIG_ValueError,"Expected CompuCell.Point3D object."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
                      
        }
         
    }
}


%typemap(in) CompuCell3D::Point3D &  (CompuCell3D::Point3D pt)  { // note that (CompuCell3D::Point3D pt) causes pt to be allocated on the stack - no need to worry abuot freeing memory
  /* Check if is a list */
    if (PyList_Check($input)) {
        int size = PyList_Size($input);        
        if (size==3){
            // CompuCell3D::Point3D pt;    
            pt.x= (short)PyInt_AsLong(PyList_GetItem($input,0));
            pt.y=(short)PyInt_AsLong(PyList_GetItem($input,1));
            pt.z=(short)PyInt_AsLong(PyList_GetItem($input,2));
            $1=&pt;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case

        }

    }else if (PyTuple_Check($input)){
        //check if it is a tuple
        int size = PyTuple_Size($input);        
        if (size==3){
            // CompuCell3D::Point3D pt;    
            pt.x= (short)PyInt_AsLong(PyTuple_GetItem($input,0));
            pt.y=(short)PyInt_AsLong(PyTuple_GetItem($input,1));
            pt.z=(short)PyInt_AsLong(PyTuple_GetItem($input,2));
            $1=&pt;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
        }                
    }else{
        
         int res = SWIG_ConvertPtr($input,(void **) &$1, $1_descriptor,0);
         
         
        if (SWIG_IsOK(res)) {
            // CompuCell3D::Point3D pt;    
            pt.x=(short)PyInt_AsLong(PyObject_GetAttrString($input,"x"));
            pt.y=(short)PyInt_AsLong(PyObject_GetAttrString($input,"y"));
            pt.z=(short)PyInt_AsLong(PyObject_GetAttrString($input,"z"));
            $1=&pt;
        } else {
        
            SWIG_exception(SWIG_ValueError,"Expected CompuCell.Point3D object."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
                      
        }
         
    }
}

// we dont really need this
// %typemap(in) CompuCell3D::Point3D *  (CompuCell3D::Point3D pt) = CompuCell3D::Point3D &  (CompuCell3D::Point3D pt);




// // // // typemap(in) CompuCell3D::Dim3D will not overshadow earlier default conversion of list to std::vector
// // // %typemap(in) CompuCell3D::Dim3D  {
  // // // /* Check if is a list */
    // // // if (PyList_Check($input)) {
        // // // int size = PyList_Size($input);        
        // // // if (size==3){
            // // // CompuCell3D::Dim3D dim;    
            // // // dim.x= (short)PyInt_AsLong(PyList_GetItem($input,0));
            // // // dim.y=(short)PyInt_AsLong(PyList_GetItem($input,1));
            // // // dim.z=(short)PyInt_AsLong(PyList_GetItem($input,2));
            // // // $1=dim;
        // // // }else{
            // // // SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case

        // // // }

    // // // }else if (PyTuple_Check($input)){
        // // // //check if it is a tuple
        // // // int size = PyTuple_Size($input);        
        // // // if (size==3){
            // // // CompuCell3D::Dim3D dim;    
            // // // dim.x= (short)PyInt_AsLong(PyTuple_GetItem($input,0));
            // // // dim.y=(short)PyInt_AsLong(PyTuple_GetItem($input,1));
            // // // dim.z=(short)PyInt_AsLong(PyTuple_GetItem($input,2));
            // // // $1=dim;
        // // // }else{
            // // // SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
        // // // }                
    // // // }else{
        
         // // // int res = SWIG_ConvertPtr($input,(void **) &$1, $&1_descriptor,0);
         
         
        // // // if (SWIG_IsOK(res)) {
            // // // CompuCell3D::Dim3D dim;    
            // // // dim.x=(short)PyInt_AsLong(PyObject_GetAttrString($input,"x"));
            // // // dim.y=(short)PyInt_AsLong(PyObject_GetAttrString($input,"y"));
            // // // dim.z=(short)PyInt_AsLong(PyObject_GetAttrString($input,"z"));
            // // // $1=dim;
        // // // } else {
        
            // // // SWIG_exception(SWIG_ValueError,"Expected CompuCell.Dim3D object."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
            
          
        // // // }
         
    // // // }
// // // }




// // // %typemap(in) CompuCell3D::Point3D  {
  // // // /* Check if is a list */
    // // // if (PyList_Check($input)) {
        // // // int size = PyList_Size($input);        
        // // // if (size==3){
            // // // CompuCell3D::Point3D pt;    
            // // // pt.x= (short)PyInt_AsLong(PyList_GetItem($input,0));
            // // // pt.y=(short)PyInt_AsLong(PyList_GetItem($input,1));
            // // // pt.z=(short)PyInt_AsLong(PyList_GetItem($input,2));
            // // // $1=pt;
        // // // }else{
            // // // SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case

        // // // }

    // // // }else if (PyTuple_Check($input)){
        // // // //check if it is a tuple
        // // // int size = PyTuple_Size($input);        
        // // // if (size==3){
            // // // CompuCell3D::Point3D pt;    
            // // // pt.x= (short)PyInt_AsLong(PyTuple_GetItem($input,0));
            // // // pt.y=(short)PyInt_AsLong(PyTuple_GetItem($input,1));
            // // // pt.z=(short)PyInt_AsLong(PyTuple_GetItem($input,2));
            // // // $1=pt;
        // // // }else{
            // // // SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
        // // // }                
    // // // }else{
        
         // // // int res = SWIG_ConvertPtr($input,(void **) &$1, $&1_descriptor,0);
         
         
        // // // if (SWIG_IsOK(res)) {
            // // // CompuCell3D::Point3D pt;    
            // // // pt.x=(short)PyInt_AsLong(PyObject_GetAttrString($input,"x"));
            // // // pt.y=(short)PyInt_AsLong(PyObject_GetAttrString($input,"y"));
            // // // pt.z=(short)PyInt_AsLong(PyObject_GetAttrString($input,"z"));
            // // // $1=pt;
        // // // } else {
        
            // // // SWIG_exception(SWIG_ValueError,"Expected CompuCell.Point3D object."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
                      
        // // // }
         
    // // // }
// // // }

