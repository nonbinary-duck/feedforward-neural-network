#pragma once
#ifndef H_530093_SRC_NEURON
#define H_530093_SRC_NEURON 1

#include <cmath>

namespace ai_assignment::activation_functions
{
    inline double stepFunc(double i)
    {
        return (i >= 0);
    }

    inline double tanhFunc(double i)
    {
        return ( std::exp(i) - std::exp(-i) ) / ( std::exp(i) + std::exp(-i) );
    }

    inline double sigmoidFunc(double i)
    {
        return 1.0l / ( 1.0l + std::exp(-i) );
    }

} // End namespace ai_assignment::activation_functions


#endif // H_530093_SRC_NEURON