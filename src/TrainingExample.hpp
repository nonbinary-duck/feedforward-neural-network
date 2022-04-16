#pragma once
#ifndef H_530093_SRC_TRAINING_EXAMPLE
#define H_530093_SRC_TRAINING_EXAMPLE 1

#include <vector>


namespace ai_assignment
{
    /**
     * @brief A data type to hold information on a training example
     */
    struct TrainingExample
    {
        std::vector<double> inputs;
        double targetOutput;
    };    

} // End namespace ai_assignment


#endif // H_530093_SRC_TRAINING_EXAMPLE