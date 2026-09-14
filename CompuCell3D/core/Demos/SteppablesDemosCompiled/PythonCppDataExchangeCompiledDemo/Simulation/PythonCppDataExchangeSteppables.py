from cc3d.core.PySteppables import SteppableBasePy


class PythonExchangeSteppable(SteppableBasePy):
    def start(self):
        self.shared_steppable_vars["python_status"] = "started"
        self.shared_steppable_vars["python_to_cpp"] = 1.0

        for cell in self.cell_list:
            cell.dict["python_signal"] = float(cell.id)
            cell.dict["python_vector"] = (float(cell.id), 10.0, 20.0)

    def step(self, mcs):
        self.shared_steppable_vars["python_to_cpp"] = float(mcs)

        cpp_value = self.shared_steppable_vars.get("cpp_to_python")
        cpp_count = self.shared_steppable_vars.get("cpp_step_count")
        for cell in self.cell_list:
            cell.dict["python_signal"] = float(mcs) + float(cell.id)
            cpp_signal = cell.dict.get("cpp_seen_signal")
            if mcs % 10 == 0:
                print("Python exchange mcs=", mcs,
                      "cell=", cell.id,
                      "cpp_signal=", cpp_signal,
                      "cpp_to_python=", cpp_value,
                      "cpp_step_count=", cpp_count)
