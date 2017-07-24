#include "osxHelper.h"
#include <stdio.h>
#include <stdlib.h>
extern "C"{
void cpp_disableGLHiDPI( long a_id ){
 printf("cpp_disableGLHiDPI %ld\n",a_id);    
 disableGLHiDPI(a_id);
 // NSView* view = reinterpret_cast<NSView*>( a_id );
 // [view setWantsBestResolutionOpenGLSurface:NO];
}


void cpp_print_hello(){
    printf(" CPP DUPA\n");
    print_hello();
 // NSLog(@"Hello, World! \n");
}

void cpp_print_hello_long( long a){
    printf(" CPPLONG %ld\n", a);
    print_hello();
 // NSLog(@"Hello, World! \n");
}



// char *
// test_get_data(unsigned int len)
// {
//     return malloc(len);
// }
//
// char *
// test_get_data_nulls(int *len)
// {
//     printf("INSIDE test_get_data_nulls");
//     *len = 5;
//     char *d = malloc(5);
//     d[0] = 'a';
//     d[1] = 'b';
//     d[2] = '\0';
//     d[3] = 'c';
//     d[4] = '\0';
//     return d;
// }
//
// void
// test_data_print(char *data, int len)
// {
//     int i;
//     for (i = 0; i < len; i++)
//         printf("%x (%c),",data[i],data[i]);
//     printf("\n");
// }

void
test_data_print_1( int len)
{
    printf("TEST DATA PRINT 1\n");
}

void
test_data_print_2()
{
    printf("TEST DATA PRINT 2\n");
}

void cpp_print_hello_1( ){
    printf("DUPA\n");
 // print_hello();
 // NSLog(@"Hello, World! \n");
}


}


// void
// test_get_data_nulls_out(char **data, int *len)
// {
//     *data = test_get_data_nulls(len);
// }
//
// void
// test_get_fixed_array_size_2(double *data)
// {
//     data[0] = 1.0;
//     data[1] = 2.0;
// }
