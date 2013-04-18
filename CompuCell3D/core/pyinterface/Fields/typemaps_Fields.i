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
  cerr<<"inside point3D conversion typemap"<<endl;
    if (PyList_Check($input)) {
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
        if (size==3){            
            dim.x= (short)PyInt_AsLong(PyTuple_GetItem($input,0));
            dim.y=(short)PyInt_AsLong(PyTuple_GetItem($input,1));
            dim.z=(short)PyInt_AsLong(PyTuple_GetItem($input,2));
            $1=&dim;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
        }                
    }else{
        
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

%typemap(in) CompuCell3D::Dim3D *  (CompuCell3D::Dim3D dim) = CompuCell3D::Dim3D &  (CompuCell3D::Dim3D dim);



%typemap(in) CompuCell3D::Point3D  (CompuCell3D::Point3D pt)  {
  /* Check if is a list */
  cerr<<"inside point3D conversion typemap"<<endl;
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
    }else if(PyArray_Check($input)){
    
        if (PyArray_NDIM($input)!=1 || PyArray_DIM($input,0)!=3){ // checking if the argument is a vector with 3 values
            SWIG_exception(SWIG_ValueError,"Expected a list/numpy array of 3 double values e.g. [12,31,48].");
        }
        
        if (PyArray_ISFLOAT ($input)){
            double * arrayContainerPtr=(double *) PyArray_DATA($input);
            pt.x=(short)arrayContainerPtr[0];
            pt.y=(short)arrayContainerPtr[1];
            pt.z=(short)arrayContainerPtr[2];
        
            
        }  else  if (PyArray_ISINTEGER ($input)){

            int * arrayContainerPtr=(int *) PyArray_DATA($input);
            pt.x=(short)arrayContainerPtr[0];
            pt.y=(short)arrayContainerPtr[1];
            pt.z=(short)arrayContainerPtr[2];

        
        }else{
            SWIG_exception(SWIG_ValueError,"The values in the array should be either floating point numbers or inttegers. Please use explicit type conversion for all the values");
        }
        
        $1=pt;        
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
  cerr<<"inside point3D conversion typemap"<<endl;
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
    }else if(PyArray_Check($input)){
    
        if (PyArray_NDIM($input)!=1 || PyArray_DIM($input,0)!=3){ // checking if the argument is a vector with 3 values
            SWIG_exception(SWIG_ValueError,"Expected a list/numpy array of 3 double values e.g. [12,31,48].");
        }
        
        if (PyArray_ISFLOAT ($input)){
            double * arrayContainerPtr=(double *) PyArray_DATA($input);
            pt.x=(short)arrayContainerPtr[0];
            pt.y=(short)arrayContainerPtr[1];
            pt.z=(short)arrayContainerPtr[2];
        
            
        }  else  if (PyArray_ISINTEGER ($input)){

            int * arrayContainerPtr=(int *) PyArray_DATA($input);
            pt.x=(short)arrayContainerPtr[0];
            pt.y=(short)arrayContainerPtr[1];
            pt.z=(short)arrayContainerPtr[2];

        
        }else{
            SWIG_exception(SWIG_ValueError,"The values in the array should be either floating point numbers or inttegers. Please use explicit type conversion for all the values");
        }
        
        
        
        
        $1=&pt;        
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

%typemap(in) CompuCell3D::Point3D *  (CompuCell3D::Point3D pt) = CompuCell3D::Point3D &  (CompuCell3D::Point3D pt);

// %typemap(in) CompuCell3D::Point3D  {
  // /* Check if is a list */
  // cerr<<"inside point3D conversion typemap"<<endl;
    // if (PyList_Check($input)) {
        // int size = PyList_Size($input);        
        // if (size==3){
            // CompuCell3D::Point3D pt;    
            // pt.x= (short)PyInt_AsLong(PyList_GetItem($input,0));
            // pt.y=(short)PyInt_AsLong(PyList_GetItem($input,1));
            // pt.z=(short)PyInt_AsLong(PyList_GetItem($input,2));
            // $1=pt;
        // }else{
            // SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case

        // }

    // }else if (PyTuple_Check($input)){
        // //check if it is a tuple
        // int size = PyTuple_Size($input);        
        // if (size==3){
            // CompuCell3D::Point3D pt;    
            // pt.x= (short)PyInt_AsLong(PyTuple_GetItem($input,0));
            // pt.y=(short)PyInt_AsLong(PyTuple_GetItem($input,1));
            // pt.z=(short)PyInt_AsLong(PyTuple_GetItem($input,2));
            // $1=pt;
        // }else{
            // SWIG_exception(SWIG_ValueError,"Expected a list of 3 integer values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
        // }                
    // }else{
        
         // int res = SWIG_ConvertPtr($input,(void **) &$1, $&1_descriptor,0);
         
         
        // if (SWIG_IsOK(res)) {
            // CompuCell3D::Point3D pt;    
            // pt.x=(short)PyInt_AsLong(PyObject_GetAttrString($input,"x"));
            // pt.y=(short)PyInt_AsLong(PyObject_GetAttrString($input,"y"));
            // pt.z=(short)PyInt_AsLong(PyObject_GetAttrString($input,"z"));
            // $1=pt;
        // } else {
        
            // SWIG_exception(SWIG_ValueError,"Expected CompuCell.Point3D object."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
                      
        // }
         
    // }
// }



