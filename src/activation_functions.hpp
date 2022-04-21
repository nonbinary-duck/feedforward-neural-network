#pragma once
#ifndef H_530093_SRC_ACTIVATION_FUNCTOINS
#define H_530093_SRC_ACTIVATION_FUNCTOINS 1

#include <cmath>

/**
 * @brief Activation functions
 */
namespace ai_assignment::activation_functions
{
    inline double stepFunc(const double &net)
    {
        return (net >= 0.0l);
    }

    inline double tanhFunc(const double &net)
    {
        //  eⁿ - e⁻ⁿ
        // ─────────
        //  eⁿ + e⁻ⁿ

        return ( std::exp(net) - std::exp(-net) ) / ( std::exp(net) + std::exp(-net) );
    }

    inline double sigmoidFunc(const double &net)
    {
        return 1.0l / ( 1.0l + std::exp(-net) );
    }

    inline double noFunc(const double &net)
    {
        return net;
    }

} // End namespace ai_assignment::activation_functions


#endif // H_530093_SRC_ACTIVATION_FUNCTOINS