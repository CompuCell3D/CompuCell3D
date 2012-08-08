import getopt
import sys
from random import randint

argv=sys.argv[1:]
opts=None
args=None

row_size=3
i_max=None
min_width=3
random_ratio=2
filename="foamCoarsening.piff"
PIFfile=None

def usage():
    print "./FoamInit.Py -r<row_size> -i<number of rows> -o<PIF file name> -z<random ratio> -m<min_width>\n"
    print "Example  ./FoamInit.py -r5 -i20 -ofoaminit2D.piff -z2 -m5\n";


try:
    opts, args = getopt.getopt(argv, "r:i:m:z:o:", ["help"])
    # print "opts=",opts
    # print "args=",args
except getopt.GetoptError, err:
    print str(err) # will print something like "option -a not recognized"
    # self.usage()
    sys.exit(2)
    
# processing command line

for o, a in opts:
    # print "o=",o
    # print "a=",a
    if o in ("-r"):
        row_size=int(a)       
    elif o in ("--help"):
        usage()
        sys.exit()
    elif o in ("-i"):
        i_max=int(a)
    elif o in ("-m"):    
        min_width=int(a)     
    elif o in ("-z"):    
        random_ratio=float(a)     
    elif o in ("-o"):    
        filename=a     
        PIFfile = file(filename,'w');

   

i=0
y_0=0
cell_counter=0;

x_max=row_size*i_max

x_max_current=0.0
y_max=0.0
z_max=0.0

for i in range(i_max) :

    x_min_current=0
    y_min=y_0+i*row_size
    z_min=0
    x_max_current=0
    y_max=y_min+row_size
    z_max=0

    row_finished_flag=False
    
    while not row_finished_flag:
        x_min_current=x_max_current
        random_width=min_width+randint(0,random_ratio*row_size)#choosing random width with min width=$_minwidth max width=$random_ratio*$row_size
        x_max_current=x_min_current+random_width

        if x_max_current>x_max:
            row_finished_flag=1
            x_max_current=x_max
            PIFfile.write('%i %s %i %i %i %i %i %i\n'%(cell_counter,'Foam',x_min_current,x_max_current,y_min,y_max, z_min ,z_max))
            # print FILE "$cell_counter ".$typeName[0]." $x_min_current $x_max_current $y_min $y_max $z_min $z_max \n";
            cell_counter+=1
        else:
            PIFfile.write('%i %s %i %i %i %i %i %i\n'%(cell_counter,'Foam',x_min_current,x_max_current,y_min,y_max, z_min ,z_max))
            # print FILE "$cell_counter ".$typeName[0]." $x_min_current $x_max_current $y_min $y_max $z_min $z_max \n";
            cell_counter+=1
    
    
    


print "Lattice dimension: x_max=",x_max_current+1," y_max=",y_max+1," z_max=",z_max+1,"\n";

    
    

# CellSize = 5;
# Height = 1;
# Length = 20;
# Width = 20;

# PIFfile = file('Foam_1.piff','w');

# CellNumber = 0;

# for xCell in arange(1,Length+1):
	# for yCell in arange(1,Width+1):
		# CellNumber = (xCell-1)*Length + yCell;
		# print xCell, yCell, CellNumber
		# xLow = (xCell-1)*CellSize+1;
		# xHigh = xLow+CellSize-1;
		# yLow = (yCell-1)*CellSize+1;
		# yHigh = yLow+CellSize-1;
		# PIFfile.write('%i %s %i %i %i %i %i %i\n'%(CellNumber,'Foam',xLow,xHigh,yLow,yHigh,0,0));
	
# PIFfile.close()
