#include <Cocoa/Cocoa.h>
#include "osxHelper.h"

void disableGLHiDPI( long a_id ){
    // NSLog(@"INT_MIN:    %i",   INT_MIN);
    //   NSLog(@"INT_MAX:    %i",   INT_MAX);
    //   NSLog(@"LONG_MIN:   %li",  LONG_MIN);    // signed long int
    //   NSLog(@"LONG_MAX:   %li",  LONG_MAX);
    //   NSLog(@"ULONG_MAX:  %lu",  ULONG_MAX);   // unsigned long int
    //   NSLog(@"LLONG_MIN:  %lli", LLONG_MIN);   // signed long long int
    //   NSLog(@"LLONG_MAX:  %lli", LLONG_MAX);
    //   NSLog(@"ULLONG_MAX: %llu", ULLONG_MAX);  // unsigned long long int

    NSLog(@"INSIDE disableGLHiDPI DUPA \n");
 NSView* view = reinterpret_cast<NSView*>( a_id );
 NSLog(@"after reinterpretcast %li \n",a_id);
 [view setWantsBestResolutionOpenGLSurface:NO];
  NSLog(@"setWantsBestResolutionOpenGLSurface:NO \n");
}


void print_hello(  ){
 NSLog(@"Hello, World! \n");
}
