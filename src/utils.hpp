#pragma once
#ifndef H_530093_SRC_UTILS
#define H_530093_SRC_UTILS 1

#include <vector>
#include <iostream>

#include "Neuron.hpp"


namespace ai_assignment::utils
{
    /**
     * @brief Releases the values inside a vector from the heap. This does nothing to the vector itself
     * 
     * @tparam T The base type of the vector (not a pointer type)
     * @param vec The vector which stores the pointers
     */
    template<typename T>
    inline void releaseVecValues(std::vector<T*> &vec)
    {
        // Iterate over the pointers and remove them
        for (size_t i = 0; i < vec.size(); i++)
        {
            delete vec[i];
        }
    }

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