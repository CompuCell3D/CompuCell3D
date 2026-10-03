from cc3d.core.PySteppables import SteppableBasePy


class SharedNumpyFieldsSteppable(SteppableBasePy):
    def step(self, mcs):
        scalar = self.field.cpp_scalar
        integer = self.field.cpp_int16
        vector = self.field.cpp_vector

        if mcs == 0:
            scalar[15, 25, 0] = 12.5
            integer[15, 25, 0] = 16
            vector[15, 25, 0, ...] = [4.0, 5.0, 6.0]

        if mcs % 10 == 0:
            print("Python shared fields mcs=", mcs,
                  "scalar=", scalar[15, 25, 0],
                  "int16=", integer[15, 25, 0],
                  "vector=", vector[15, 25, 0, ...])
