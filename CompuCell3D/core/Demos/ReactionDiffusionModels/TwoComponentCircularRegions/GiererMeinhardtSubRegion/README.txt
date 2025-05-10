This simulation shows the spatial pattern formation in reaction diffusion systems using Gierer-Meinhardt model exhibiting the Turing type instability on circular regions. 

Expected Result: The simulation should exhibit spatially inhomogeneous, temporally stable patterns under the proper selection of system parameters on circular regions. The results should be similar to that of the Gierer-Meinhardt version 2 model, but the substrate should be contained within a circular region.  

The evolution of self-organizing pattern formation was introduced by Alan Turing [1] providing a detailed explanation of how reaction-diffusion systems could be responsible for the emergence of pattern formation in nature. Gierer-Meinhardt system is another example of a model showing the Turing type of patterns. It was introduced by Alfred Gierer and Hans Meinhardt in 1972 [2]. Gierer and Meinhardt have proved that a short-range activation and long-range inhibition mechanism is a requirement for such patterns to evolve. Mathematically, this means that the diffusion coefficient of the inhibitor is significantly larger than the diffusion coefficient of the activator [2,3]. One can see how diffusion coefficients are selected as well as how reaction functions emerge as an additional term in the XML file. We note that, change in any of the parameters can lead to different types of patterns. To obtain the circular region we use the frozen cells. This can be performed for the frozen cells by adding a new CellType to the CellType plugin and indicating them in the BlobInitializer section together with the other cell types. This way of simulation resemble a petri dish experiment, thus CompuCell3D effectively illustrates the emergence of labyrinthine type of patterns.


References:

[1] Turing, A. M. (1990). The chemical basis of morphogenesis. Bulletin of mathematical biology, 52, 153-197.
[2] Gierer A., Meinhardt H. (1972) A theory of biological pattern formation, Kybernetik 12: 30–39.
[3] Murray J.D. (2001). Mathematical biology II: Spatial models and biomedical applications, Vol. 3, Springer New York.



Author: Gülsemay YİĞİT, 2024
