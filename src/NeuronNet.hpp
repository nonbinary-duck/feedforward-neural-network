#pragma once
#ifndef H_530093_SRC_NET
#define H_530093_SRC_NET 1

#include <mutex>
#include <vector>

#include "Connection.hpp"
#include "Neuron.hpp"
#include "utils.hpp"


namespace ai_assignment
{
    // Make the code more readable, scope the statement to not interfere with future libraries
    using std::vector;
    
    /**
     * @brief A network of artifical neurons, thread safe
     */
    class NeuronNet
    {
        public:

            // Constructors


            /**
             * @brief Construct a new Neuron Net according to some patterns. Does not properly verify the net structure and will produce undefined behaviour if it's invalid
             * 
             * @param netArchitecture The layout of the neurons. Each element represents the number of neurons in that layer
             * @param inputArchitecture The number of inputs in each layer of the net. Must include bias/threshold input.
             * @param inputs The number of inputs accepted from each neuron, including the bias/threshold 'fake input'. If a layer of the architecture doesn't have enough inputs from the previous layer, it takes the first value of the layer before that. i.e. the very first input can be used as a 'fallback' bias/threshold
             * @param activationFunctions The activation function to use for each individual layer
             * @param startingWeights The weights to apply to each neuron. A weight (l) set of weights (k*) is part of a neuron (j) which is part of a layer (i). Auto-generates weights if nullptr. WARNING: This needs to be on the heap. It's disposed immediately after the neurons have been created
             */
            NeuronNet(
                vector<size_t> netArchitecture,
                const vector<size_t> inputArchitecture,
                const size_t inputs,
                const vector< activation_func_type > activationFunctions,
                vector< vector < vector< double >* > > *startingWeights = nullptr
            );

            /**
             * @brief Temporary shallow copy ctor, not valid
             * 
             * @note We don't copy the mutex since it needs to be reset
             * 
             * @param obj object to copy
             */
            inline NeuronNet(const NeuronNet &obj) noexcept
                : m_Architecture(obj.m_Architecture),
                    m_ConnectionHeuristic(obj.m_ConnectionHeuristic),
                    m_InputArchitecture(obj.m_InputArchitecture)
            {}

            /**
             * @brief Destroy the NeuronNet object
             */
            inline virtual ~NeuronNet() noexcept
            {
                utils::releaseVecValues<vector<double>>(this->m_ConnectionHeuristic);

                for (size_t i = 0; i < this->m_Architecture.size(); i++)
                {
                    utils::releaseVecValues<Neuron>(this->m_Architecture[i]);
                }
            }


            // Functions


            /**
             * @brief Runs through the net and returns the results
             * 
             * @param inputLayers Input values. Must be a vector of non-zero size containing vectors of exactly inputCount + 1 size
             * @return double The result from going through the entire net
             */
            vector<double> ProcessInputs(vector<vector<double>*> &inputLayers);

        protected:

            // Properties


            /**
             * @brief The architecture of the net
             */
            vector<vector<Neuron*>> m_Architecture;

            /**
             * @brief A list of layers of values to connect the neurons. Size of inputCount + 1 repeated n times where n > 2
             */
            vector<vector<double>*> m_ConnectionHeuristic;

            /**
             * @brief The number of inputs given layer, includes bias/threshold
             */
            const vector<size_t> m_InputArchitecture;
        
        private:

            /**
             * @brief A mutex to guard m_ConnectionHeuristic and m_Architecture
             */
            mutable std::mutex m_Lock;
    };
    
} // End namespace ai_assignment


#endif // H_530093_SRC_NET