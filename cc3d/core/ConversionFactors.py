class ConversionFactors:
    def __init__(self):
        self.time_unit = ""
        self.time_scaling_factor = 1.0
        self.length_unit = ""
        self.length_scaling_factor = 1.0

    def __iter__(self):
        """
        supports this type of unpacking
        c_factors = ConversionFactors()
        time_unit, time_scaling_factor, length_unit, length_scaling_factor = c_factors
        :return:
        """
        yield self.time_unit
        yield self.time_scaling_factor
        yield self.length_unit
        yield self.length_scaling_factor