from cc3d.cpp import CompuCell
from cc3d.core.PySteppables import SteppableBasePy


class GeometryComparisonSteppable(SteppableBasePy):
    def start(self):
        p1 = (10.25, 12.5, 0.0)
        p2 = (95.75, 70.0, 0.0)
        ordinary = self.distance_vector(p1, p2)
        invariant = self.invariant_distance_vector(p1, p2)
        unconditional = self.unconditional_invariant_distance_vector(p1, p2)
        hex_point = self.hex_2_cartesian((1.0, 2.0, 0.0))
        cart_point = self.cartesian_2_hex((3, 4, 0))

        shared = self.shared_steppable_vars
        shared["geometry_python_distance"] = self.distance(p1, p2)
        shared["geometry_python_ordinary_vector"] = tuple(float(v) for v in ordinary)
        shared["geometry_python_invariant_distance"] = self.vector_norm(invariant)
        shared["geometry_python_invariant_vector"] = tuple(float(v) for v in invariant)
        shared["geometry_python_unconditional_vector"] = tuple(float(v) for v in unconditional)
        shared["geometry_python_norm"] = self.vector_norm((3.0, 4.0, 12.0))
        shared["geometry_python_hex_point"] = (float(hex_point.x), float(hex_point.y), float(hex_point.z))
        shared["geometry_python_cart_point"] = (float(cart_point.x), float(cart_point.y), float(cart_point.z))
        shared["geometry_python_in_bounds"] = bool(
            self.simulator.getBoundaryStrategy().isValid(CompuCell.Point3D(10, 10, 0))
        )

    def step(self, mcs):
        shared = self.shared_steppable_vars
        if "geometry_cpp_failures" not in shared:
            return
        failures = shared.get("geometry_cpp_failures", -1)
        print("Geometry ABI comparison mcs=", mcs, "failures=", failures,
              "max_error=", shared.get("geometry_cpp_max_error"))
