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
     * @brief A network of artifical neurons, thread safe. Bias/threshold is final value of input
     */
    class NeuralNet
    {
        public:

            // Constructors


            /**
             * @brief Construct a new Neuron Net according to some patterns. Does not properly verify the net structure and will produce undefined behaviour if it's invalid
             * 
             * @param netArchitecture The layout of the neurons. Each element represents the number of neurons in that layer
             * @param inputArchitecture The number of inputs in each layer of the net. Must include bias/threshold input. Each number is the count of inputs in that layer, including the bias/threshold 'fake input'. If a layer of the architecture doesn't have enough inputs from the previous layer, it takes the last value of the layer before that. i.e. the very last input can be used as a 'fallback' bias/threshold
             * @param activationFunctions The activation function to use for each individual layer
             * @param startingWeights The weights to apply to each neuron. Must contain every single weight. A weight (l) set of weights (k*) is part of a neuron (j) which is part of a layer (i). Auto-generates weights if nullptr. WARNING: This needs to be on the heap. It's disposed immediately after the neurons have been created. The neuron has ownership of the nested heap value, which is released when the neuron gets deleted when the NeuralNet gets freed
             */
            NeuralNet(
                const vector<size_t> netArchitecture,
                const vector<size_t> inputArchitecture,
                const vector< activation_func_type > activationFunctions,
                vector< vector < vector< double >* > > *startingWeights = nullptr
            );

            /**
             * @brief The copy constructor
             * 
             * @note We don't copy the mutex since it needs to be reset, nor do we copy the heuristics since it needs to be completely re-created
             * 
             * @param obj object to copy
             */
            NeuralNet(const NeuralNet &obj) noexcept;

            /**
             * @brief Destroy the NeuralNet object
             */
            inline virtual ~NeuralNet() noexcept
            {
                utils::releaseVecValues<double>(this->m_ConnectionHeuristic);
                utils::releaseVecValues<Neuron>(this->m_Architecture);
            }


            // Functions


            /**
             * @brief Runs through the net and returns the results
             * 
             * @param inputLayers Input values. Must be a vector of non-zero size containing vectors of exactly inputCount + 1 size
             * @return double The results from the final layer of the network
             */
            vector<double> ProcessInputs(vector<vector<double>*> &inputLayers);

        protected:

            // Properties


            /**
             * @brief The architecture of the net
             */
            vector<vector<Neuron*>> m_Architecture;

            /**
             * @brief A list of layers of values to connect the neurons. The first layer is the input layer. In each layer, the last part is dedicated to inputs and/or fallback inputs
             */
            vector<vector<double*>> m_ConnectionHeuristic;

            /**
             * @brief The number of inputs given for each layer, includes bias/threshold
             */
            const vector<size_t> m_InputArchitecture;

            /**
             * @brief The architecture of the net, cached to speed up copy ctor
             */
            const vector<size_t> m_NetArchitecture;

            /**
             * @brief A mutex to guard m_ConnectionHeuristic and m_Architecture
             */
            mutable std::mutex m_Lock;


            // Functions


            /**
             * @brief Initialises the heuristic table
             */
            void InitialiseConnectionHeuristics(
                const vector<size_t> &netArchitecture,
                const vector<size_t> &inputArchitecture
            ) noexcept;

            /**
             * @brief Initialise the neurons
             */
            void InitialiseNeurons(
                const vector<size_t> &netArchitecture,
                const vector<size_t> &inputArchitecture,
                const vector< activation_func_type > &activationFunctions,
                vector< vector < vector< double >* > > *startingWeights = nullptr
            );
    };
    
} // End namespace ai_assignment


#endif // H_530093_SRC_NET