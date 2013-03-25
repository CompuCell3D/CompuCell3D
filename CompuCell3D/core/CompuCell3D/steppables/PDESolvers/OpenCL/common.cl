
size_t d3To1d(int4 ind3, int3 dim){
	return dim.x*dim.y*ind3.z+dim.x*ind3.y+ind3.x;	
}
