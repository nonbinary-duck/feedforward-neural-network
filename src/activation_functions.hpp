#pragma once
#ifndef H_530093_SRC_NEURON
#define H_530093_SRC_NEURON 1

#include <cmath>

/**
 * @brief Activation functions
 */
namespace ai_assignment::activation_functions
{
    inline double stepFunc(double net)
    {
        return (net >= 0.0l);
    }

    inline double tanhFunc(double net)
    {
        //  eⁿ - e⁻ⁿ
        // ─────────
        //  eⁿ + e⁻ⁿ

        return ( std::exp(net) - std::exp(-net) ) / ( std::exp(net) + std::exp(-net) );
    }

    inline double sigmoidFunc(double net)
    {
        return 1.0l / ( 1.0l + std::exp(-net) );
    }

} // End namespace ai_assignment::activation_functions


#endif // H_530093_SRC_NEURON