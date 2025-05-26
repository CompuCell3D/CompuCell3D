%define FIELD3DEXTENDERBASE(className,returnType)
%extend  className{

        std::string __str__(){
            std::ostringstream s;
            s <<#className << " dim" << self->getDim();
            return s.str();
        }

        returnType min(){
            returnType minVal = self->get(Point3D(0, 0, 0));

            Dim3D dim = self->getDim();

            for (int x = 0; x < dim.x; ++x)
                for (int y = 0; y < dim.y; ++y)
                    for (int z = 0; z < dim.z; ++z) {
                        returnType val = self->get(Point3D(x, y, z));
                        if (val < minVal) minVal = val;
                    }

            return minVal;

        }

        returnType max(){
            returnType maxVal = self->get(Point3D(0, 0, 0));

            Dim3D dim = self->getDim();

            for (int x = 0; x < dim.x; ++x)
                for (int y = 0; y < dim.y; ++y)
                    for (int z = 0; z < dim.z; ++z) {
                        returnType val = self->get(Point3D(x, y, z));
                        if (val > maxVal) maxVal = val;
                    }

            return maxVal;

        }

        returnType __getitem__(PyObject *_indexTuple) {
            if (!PyTuple_Check(_indexTuple) || PyTuple_GET_SIZE(_indexTuple) != 3) {
                throw
                std::runtime_error(std::string(#className)+std::string(
                        ": Wrong Syntax: Expected something like: field[1,2,3]"));
            }
            PyObject *xCoord = PyTuple_GetItem(_indexTuple, 0);
            PyObject *yCoord = PyTuple_GetItem(_indexTuple, 1);
            PyObject *zCoord = PyTuple_GetItem(_indexTuple, 2);
            Py_ssize_t x, y, z;

            //x-coord
            if (PyInt_Check(xCoord)) {
                x = PyInt_AsLong(PyTuple_GetItem(_indexTuple, 0));
            } else if (PyLong_Check(xCoord)) {
                x = PyLong_AsLong(PyTuple_GetItem(_indexTuple, 0));
            } else if (PyFloat_Check(xCoord)) {
                x = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple, 0)));
            } else {
                throw
                std::runtime_error(
                        "Wrong Type (X): only integer or float values are allowed here - floats are rounded");
            }
            //y-coord
            if (PyInt_Check(yCoord)) {
                y = PyInt_AsLong(PyTuple_GetItem(_indexTuple, 1));
            } else if (PyLong_Check(yCoord)) {
                y = PyLong_AsLong(PyTuple_GetItem(_indexTuple, 1));
            } else if (PyFloat_Check(yCoord)) {
                y = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple, 1)));
            } else {
                throw
                std::runtime_error(
                        "Wrong Type (Y): only integer or float values are allowed here - floats are rounded");
            }
            //z-coord
            if (PyInt_Check(zCoord)) {
                z = PyInt_AsLong(PyTuple_GetItem(_indexTuple, 2));
            } else if (PyLong_Check(zCoord)) {
                z = PyLong_AsLong(PyTuple_GetItem(_indexTuple, 2));
            } else if (PyFloat_Check(zCoord)) {
                z = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple, 2)));
            } else {
                throw
                std::runtime_error(
                        "Wrong Type (Z): only integer or float values are allowed here - floats are rounded");
            }

            return self->get(Point3D(x, y, z));
        }
}

%enddef



%define FIELD3DEXTENDER(className,returnType)
FIELD3DEXTENDERBASE(className,returnType)

%extend className{

    %pythoncode %{

        def normalizeSlice(self, s):
            norm = lambda x : x if x is None else int(round(x))
            return slice ( norm(s.start),norm(s.stop), norm(s.step) )

        def __setitem__(self,_indexTyple,_val):
            newSliceTuple = tuple(map(lambda x : self.normalizeSlice(x) if isinstance(x,slice) else x , _indexTyple))
            self.setitem(newSliceTuple,_val)

    %}

  void setitem(PyObject *_indexTuple,returnType _val) {
  // void __setitem__(PyObject *_indexTuple,returnType _val) {
    if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
        throw std::runtime_error("Wrong Syntax: Expected something like: field[1,2,3]=object");
    }

    PyObject *xCoord=PyTuple_GetItem(_indexTuple,0);
    PyObject *yCoord=PyTuple_GetItem(_indexTuple,1);
    PyObject *zCoord=PyTuple_GetItem(_indexTuple,2);

    Py_ssize_t  start_x, stop_x, step_x, sliceLength;
    Py_ssize_t  start_y, stop_y, step_y;
    Py_ssize_t  start_z, stop_z, step_z;

    Dim3D dim=self->getDim();

    if (PySlice_Check(xCoord)){
		int ok = PySlice_GetIndices(xCoord, dim.x, &start_x, &stop_x, &step_x);

     // cout<<"extracting slices for x axis"<<endl;
     //cerr<<"start x="<< start_x<<endl;
     //cerr<<"stop x="<< stop_x<<endl;
     //cerr<<"step x="<< step_x<<endl;
     //cerr<<"sliceLength="<<sliceLength<<endl;
     //cerr<<"ok="<<ok<<endl;

    }else{
        if (PyInt_Check(xCoord)){
            start_x=PyInt_AsLong(PyTuple_GetItem(_indexTuple,0));
            stop_x=start_x;
            step_x=1;
        }else if (PyLong_Check(xCoord)){
            start_x=PyLong_AsLong(PyTuple_GetItem(_indexTuple,0));
            stop_x=start_x;
            step_x=1;
        }else if (PyFloat_Check(xCoord)){
            start_x = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,0)));
            stop_x=start_x;
            step_x=1;
        }
        else{
            throw std::runtime_error("Wrong Type (X): only integer or float values are allowed here - floats are rounded");
        }

        start_x %= dim.x;
        stop_x %= dim.x;
        stop_x += 1;

        if (start_x < 0)
            start_x = dim.x + start_x;

        if (stop_x < 0)
            stop_x = dim.x + stop_x;

    }

    if (PySlice_Check(yCoord)){

		int ok = PySlice_GetIndices(yCoord, dim.y, &start_y, &stop_y, &step_y);


    }else{
        if (PyInt_Check(yCoord)){
            start_y=PyInt_AsLong(PyTuple_GetItem(_indexTuple,1));
            stop_y=start_y;
            step_y=1;
        }else if (PyLong_Check(yCoord)){
            start_y=PyLong_AsLong(PyTuple_GetItem(_indexTuple,1));
            stop_y=start_y;
            step_y=1;
        }else if (PyFloat_Check(yCoord)){
            start_y = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,1)));
            stop_y=start_y;
            step_y=1;
        }
        else{
            throw std::runtime_error("Wrong Type (Y): only integer or float values are allowed here - floats are rounded");
        }

        start_y %= dim.y;
        stop_y %= dim.y;
        stop_y += 1;

        if (start_y < 0)
            start_y = dim.y + start_y;

        if (stop_y < 0)
            stop_y = dim.y + stop_y;

    }

    if (PySlice_Check(zCoord)){

	   int ok = PySlice_GetIndices(zCoord, dim.z, &start_z, &stop_z, &step_z);

    }else{
        if (PyInt_Check(zCoord)){
            start_z=PyInt_AsLong(PyTuple_GetItem(_indexTuple,2));
            stop_z=start_z;
            step_z=1;
        }else if (PyLong_Check(zCoord)){
            start_z=PyLong_AsLong(PyTuple_GetItem(_indexTuple,2));
            stop_z=start_z;
            step_z=1;
        }else if (PyFloat_Check(zCoord)){
            start_z = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,2)));
            stop_z=start_z;
            step_z=1;
        }
        else{
            throw std::runtime_error("Wrong Type (Z): only integer or float values are allowed here - floats are rounded");
        }
        start_z %= dim.z;
        stop_z %= dim.z;
        stop_z += 1;

        if (start_z < 0)
            start_z = dim.z + start_z;

        if (stop_z < 0)
            stop_z = dim.z + stop_z;


    }


    PyObject *sliceX=0,*sliceY=0,* sliceZ=0;

    //cout << "start_x, stop_x = " << start_x << "," << stop_x << endl;
    //cout << "start_y, stop_y = " << start_y << "," << stop_y << endl;
    //cout << "start_z, stop_z = " << start_z << "," << stop_z << endl;
    for (Py_ssize_t x=start_x ; x<stop_x ; x+=step_x)
        for (Py_ssize_t y=start_y ; y<stop_y ; y+=step_y)
            for (Py_ssize_t z=start_z ; z<stop_z ; z+=step_z){
                $self->set(Point3D(x,y,z),_val);
            }

  }


}
%enddef
