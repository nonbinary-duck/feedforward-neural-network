#pragma once
#ifndef H_530093_SRC_UTILS
#define H_530093_SRC_UTILS 1

#include <cstdint>


namespace utils
{
    // Give neater names to the forced-width int data types
    // least type defines smallest possible type that is larger than or equal to x (i.e. in 9-bit systems, a 64 would be 72 bits wide)
    // Scoped as to not interfere with other sources
    typedef uint_least64_t    tulong;
    typedef uint_least32_t    tuint;
    typedef uint_least16_t    tushort;
    typedef uint_least8_t     tubyte;

    typedef int_least64_t     slong;
    typedef int_least32_t     sint;
    typedef int_least16_t     sshort;
    typedef int_least8_t      sbyte;
    
} // End namespace utils

#endif // H_530093_SRC_UTILS