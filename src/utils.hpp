#pragma once
#ifndef H_530093_SRC_UTILS
#define H_530093_SRC_UTILS 1

#include <iostream>

#include "Neuron.hpp"


namespace ai_assignment::utils
{

    inline void printWeights(Neuron *n)
    {
        // Fetch a copy
        auto weights = n->GetWeights();
        
        for (size_t i = 0; i < weights->size(); i++)
        {
            std::cout << "weight " << i << ": " << weights->at(i) << std::endl;
        }
        
        // Dispose
        delete weights;
    }
    
} // End namespace utils

#endif // H_530093_SRC_UTILS