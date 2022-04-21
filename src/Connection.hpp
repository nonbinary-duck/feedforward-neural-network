#pragma once
#ifndef H_530093_SRC_CONNECTION
#define H_530093_SRC_CONNECTION 1

#include <vector>

#include "Neuron.hpp"


namespace ai_assignment
{
    /**
     * @brief A single coordinate
     */
    struct Coordinate
    {
        size_t i, j;
    };
    
    
    /**
     * @brief A data type to hold information on how two things connect
     */
    struct Connection
    {
        /**
         * @brief The index of the slice of data that connects to the neuron
         */
        size_t dataSlicePos;

        /**
         * @brief The coordinate of the neuron
         */
        Coordinate neuronCoord;
    };
} // End namespace ai_assignment


#endif // H_530093_SRC_CONNECTION