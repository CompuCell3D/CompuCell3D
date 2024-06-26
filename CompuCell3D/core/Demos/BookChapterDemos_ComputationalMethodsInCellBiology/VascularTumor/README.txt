This demo is incorrect and pending fixes. The bug is that all cells generate in the same area. 

Here, Glucose bolsters cell growth and proliferation but also enables us to model cell death when it is absent. 

Expected result: A large amount of Glucose is placed into the medium during initialization because of the attached .dat file. Later, this Glucose will be consumed by the cancer cells. By the end, VEGF1 has a high concentration somewhere in the medium, somewhat far away from the cell blob. VEGF2, conversely, is high in the cell blob because it is secreted by Proliferating-type cells, although its diffusion is limited by uptake from Vascular and NeoVascular cells. 

See the included VascularTumor.xml file for more details about the calculations. 