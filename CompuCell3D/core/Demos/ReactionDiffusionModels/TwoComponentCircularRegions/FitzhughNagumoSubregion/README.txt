This simulation shows the spatial pattern formation in reaction diffusion systems using Fitzhugh-Nagumo model exhibiting the Turing type instability. 

Expected Result: The simulation should exhibit spatially inhomogeneous, temporally stable patterns under the proper selection of system parameters on circular type of regions.
 
The FitzHugh-Nagumo model is a well-known reaction-diffusion system, introduced in the study of electrical interaction of the nerves. The evolution of self-organizing pattern formation was introduced by Alan Turing [2] providing a detailed explanation of how reaction-diffusion systems could be responsible for the emergence of pattern formation in nature. For such patterns to evolve, a short-range activation and long-range inhibition mechanism is a requirement. This means that the diffusion coefficient of the inhibitor is significantly larger than the diffusion coefficient of the activator [3,4]. One can see how diffusion coefficients are selected as well as how reaction functions emerge as an additional term in the XML file. To obtain the circular region we use the frozen cells. This can be performed for the frozen cells by adding a new CellType to the CellType plugin and indicating them in the BlobInitializer section together with the other cell types. This way of simulation resemble a petri dish experiment, thus CompuCell3D effectively illustrates the emergence of elegant stripe-spot type of patterns on circular regions.

References:

[1] FitzHugh, R. (1961). Impulses and physiological states in theoretical models of nerve membrane. Biophysical journal, 1(6), 445-466.
[2] Turing, A. M. (1990). The chemical basis of morphogenesis. Bulletin of mathematical biology, 52, 153-197.
[3] Murray J.D. (2001). Mathematical biology II: Spatial models and biomedical applications, Vol. 3, Springer New York.
[4] Gierer A., Meinhardt H. (1972) A theory of biological pattern formation, Kybernetik 12: 30–39.


Author: Gülsemay YİĞİT, 2024
