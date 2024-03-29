#pragma once
#ifndef H_530093_SRC_NEURAL_NET
#define H_530093_SRC_NEURAL_NET 1

#include "NeuralNet.fwd.hpp"
#include "Neuron.fwd.hpp"

#include <cmath>
#include <mutex>
#include <vector>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "utils.hpp"
#include "Neuron.hpp"
#include "TrainingExample.hpp"


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

            // Definitions
            
            typedef TrainingExample<std::vector<double>>    Example;
            typedef vector<vector<vector<double>>>          weight_type;


            // Constructors


            /**
             * @brief Construct a new Neuron Net according to some patterns. Does not properly verify the net structure and will produce undefined behaviour if it's invalid
             * 
             * @param netArchitecture The layout of the neurons. Each element represents the number of neurons in that layer
             * @param inputArchitecture The number of inputs each neuron takes. Must include bias/threshold. Values will carry-over until a neuron overwrites them (i.e. the last value can be used as a bias/threshold)
             * @param activationFunctions The activation function to use for each individual layer
             * @param startingWeights The weights to apply to each neuron. Must contain every single weight. A weight (l) set of weights (k*) is part of a neuron (j) which is part of a layer (i). Auto-generates weights if nullptr. WARNING: This needs to be on the heap. It's disposed immediately after the neurons have been created. The neuron has ownership of the nested heap value, which is released when the neuron gets deleted when the NeuralNet gets freed
             */
            NeuralNet(
                const vector<size_t> netArchitecture,
                const size_t inputs,
                const vector< Neuron::activation_func_type > activationFunctions,
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
                utils::releaseVecValues<Neuron>(this->m_Architecture);
            }

            // Accessors

            /**
             * @brief Get a copy of all of the weights in the network
             */
            inline weight_type *GetWeights() const noexcept
            {
                auto *out = new weight_type(this->m_Architecture.size());

                for (size_t i = 0; i < out->size(); i++)
                {
                    out->at(i) = vector<vector<double>>(this->m_NetArchitecture.at(i));
                    
                    for (size_t j = 0; j < out->at(i).size(); j++)
                    {
                        // Create a copy on the "stack" of the heap of the output of each neuron's weight
                        out->at(i).at(j) = vector<double>(*this->m_Architecture.at(i).at(j)->m_Weights);
                    }
                }

                return out;
            }

            inline void SetWeights(weight_type *newWeights) const noexcept
            {
                for (size_t i = 0; i < newWeights->size(); i++)
                {
                    for (size_t j = 0; j < newWeights->at(i).size(); j++)
                    {
                        // Create a copy on the "stack" of the heap of the output of each neuron's weight
                        *this->m_Architecture.at(i).at(j)->m_Weights = newWeights->at(i).at(j);
                    }
                }
            }

            /**
             * @brief Prints the weights to a csv file stream
             */
            inline void PrintWeights(std::fstream &out) const noexcept
            {
                auto *w = this->GetWeights();

                // Each layer
                for (size_t i = 0; i < w->size(); i++)
                {
                    // Each neuron
                    for (size_t j = 0; j < w->at(i).size(); j++)
                    {
                        // Each weight
                        for (size_t k = 0; k < w->at(i).at(j).size(); k++)
                        {
                            // Amend the weight to the file
                            out << w->at(i).at(j).at(k) << ',';
                        }
                    }
                }

                out << std::endl;

                delete w;
            }

            // Functions


            /**
             * @brief Runs through the net and returns the results. Not thread safe
             * 
             * @param inputs The inputs to the net. The last value of the inputs is the bias/threshold which is in all layers until overwritten by a neuron
             * @param recordedOutputs If provided, records each individual output. This excludes the final output, and should therefore have a size of layers * inputs
             * @return double The results from the final layer of the network
             */
            vector<double> *ProcessInputs(vector<double> inputs, vector<vector<double>> *recordedOutputs = nullptr);

            /**
             * @brief Trains the neural network until the mean squared error stops changing. Thread safe
             * 
             * @param trainingExamples Examples to give the net for it to "learn"
             * @param learningRate The learning rate
             * @return The number of epochs taken to fully train the network
             */
            size_t TrainNetwork(vector<Example> &trainingExamples, double learningRate);

            /**
             * @brief Trains the neural network for one epoch, then returns the error rate. Not thread safe
             * 
             * @param trainingExample The example to give the net for it to "learn"
             * @param learningRate The learning rate
             * @param sharedOutputCache A shared variable to reduce overhead
             * @param newWeights Must be initialised to the correct size. Is set to the new values of the weights as an optimisation step over creating a new loop counter elsewhere
             * @return double The error of the net: netTarget - netOutput
             */
            double TrainNetwork(Example &trainingExample, double &learningRate, vector<vector<double>> *sharedOutputCache, weight_type *newWeights);

        protected:

            // Properties


            /**
             * @brief The neurons
             */
            vector<vector<Neuron*>> m_Architecture;

            /**
             * @brief The number of inputs each neuron takes
             */
            const size_t m_Inputs;

            /**
             * @brief The architecture of the net
             */
            const vector<size_t> m_NetArchitecture;

            /**
             * @brief A mutex to guard m_ConnectionHeuristic and m_Architecture
             */
            mutable std::mutex m_Lock;


            // Functions

            /**
             * @brief Initialise the neurons
             */
            void InitialiseNeurons(
                const vector<size_t> &netArchitecture,
                const size_t inputs,
                const vector< Neuron::activation_func_type > &activationFunctions,
                vector< vector < vector< double >* > > *startingWeights = nullptr
            );
    };
    
} // End namespace ai_assignment


#endif // H_530093_SRC_NEURAL_NET