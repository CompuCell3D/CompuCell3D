This divides the cell field into 4 quadrants to demonstrate the effect of temperature (fluctuation amplitude) on cell membrane activity. 

The Python code selects the top-left quadrant of the cell blob and assigns a high fluctuation amplitude to them. After a moment, it sets the value back to -1 and selects the next quadrant in a clockwise manner. Negative values of ``fluctuationAmplitude`` are ignored, so those cells will use the globally-defined XML value of 5.

In general, the property can be set using either FluctuationAmplitude or Temperature in XML or fluctuation_amplitude in Python. 