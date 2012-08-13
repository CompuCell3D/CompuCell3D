#!/usr/bin/perl -w

use strict;

sub usage{
    print "./FoamInit.pl -r<row_size> -i<number of rows> -o<PIF file name> -z<random ratio> -m<min_width>\n";
    print "Example  ./FoamInit.pl -r5 -i20 -ofoaminit2D.pif -z2 -m5\n";
}

my $i_max=10;
my $j_max=0;
my $k_max=0;

my $row_size=3;
my $random_ratio=2;

my $min_width;
my $got_min_width=0;
my $i=0;
my $j=0;
my $k=0;

my $x_0=0;
my $y_0=0;
my $z_0=0;
# my $gap=0;
# my $type_max=1;
# my $radius=0;

my $filename="flow_square.pif";
my @typeName=qw(Foam) ; #type name array

    foreach(@ARGV){
	    if(/^-r(.+)/){$row_size=$1;}
       if(/^-i(.+)/){$i_max=$1;}
       if(/^-m(.+)/){$min_width=$1; $got_min_width=1;}
	    if(/^-z(.+)/){$random_ratio=$1;}
	    if(/^-o(.+)/){$filename=$1;}
	    if(/^--help/){usage();exit(0);}
	    
    }


if(!$got_min_width){
   $min_width=$row_size;
}

open(FILE,">$filename");

my $cell_counter=0;
my $cell_type=1;

my $x_min=0;
my $y_min=0;
my $z_min=0;

my $x_max=0;
my $y_max=0;
my $z_max=0;
my $y_cent;
my $z_cent;


$y_cent=int($y_max/2);
$z_cent=int($z_max/2);


my $x_min_current;
my $x_max_current;
my $row_finished_flag;
my $random_width=0;

$x_max=$row_size*$i_max;

for($i=0 ; $i<$i_max ;++$i){

    $x_min_current=0;
    $y_min=$y_0+$i*($row_size);
    $z_min=0;
    $x_max_current=0;;
    $y_max=$y_min+$row_size;
    $z_max=0;

    $row_finished_flag=0;
    
    while(!$row_finished_flag){
	$x_min_current=$x_max_current;
	$random_width=$min_width+int(rand($random_ratio*$row_size)); #choosing random width with min width=$_minwidth max width=$random_ratio*$row_size
	$x_max_current=$x_min_current+$random_width;

	if($x_max_current>$x_max){
	    $row_finished_flag=1;
	    $x_max_current=$x_max;
	    
	    print FILE "$cell_counter ".$typeName[0]." $x_min_current $x_max_current $y_min $y_max $z_min $z_max \n";
	    ++$cell_counter;
	}else{
	    print FILE "$cell_counter ".$typeName[0]." $x_min_current $x_max_current $y_min $y_max $z_min $z_max \n";
	    ++$cell_counter;
	}
    }



}

print "Lattice dimension: x_max=".eval($x_max_current+1)." y_max=".eval($y_max+1)." z_max=".eval($z_max+1)."\n";
exit;

