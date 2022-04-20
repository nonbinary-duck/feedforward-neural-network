#pragma once
#ifndef H_530093_SRC_NEURON
#define H_530093_SRC_NEURON 1

#include <functional>
#include <stdexcept>
#include <random>
#include <vector>

#include "TrainingExample.hpp"


namespace ai_assignment
{
    /**
     * @brief An artificial 'neuron'
     */
    class Neuron
    {
        public:

            // Constructors


            /**
             * @brief Construct a new 'Neuron' and generate small random values to initiate the weights to (except the bias weight, which will be 1.0)
             * 
             * @param inputCount The number of inputs to the neuron, excluding the bias/threshold
             * @param activationFunction The activation function to apply to the output
             */
            Neuron(size_t inputCount, const std::function<double(const double&)> &activationFunction);

            /**
             * @brief Construct a new 'Neuron' and copy the weights into the heap
             * 
             * @param inputCount The number of inputs to the neuron, excluding the bias/threshold
             * @param weights The starting weights for the 'Neuron'; gets coppied onto the heap and must have an extra 'one' for the bias/threshold
             * @param activationFunction The activation function to apply to the output
             */
            Neuron(size_t inputCount, std::vector<double> &weights, const std::function<double(double)> &activationFunction);

            /**
             * @brief Construct a new 'Neuron'
             * 
             * @param inputCount The number of inputs to the neuron, excluding the bias/threshold
             * @param weights The starting weights for the 'Neuron'; must have an extra 'one' for the bias/threshold
             * @param activationFunction The activation function to apply to the output
             */
            Neuron(size_t inputCount, std::vector<double> *weights, const std::function<double(double)> &activationFunction);

            /**
             * @brief Copy ctor
             * 
             * @param obj object to copy
             */
            inline Neuron(const Neuron &obj) noexcept : InputCount(obj.InputCount), m_Weights(obj.m_Weights), m_ActivationFunction(obj.m_ActivationFunction)
            {}

            /**
             * @brief Destroy the Neuron object (we only own our weights on the heap)
             */
            inline ~Neuron() noexcept
            {
                delete this->m_Weights;
            }

            // Properties

            /**
             * @brief The number of inputs accepted by this 'neuron'
             */
            const size_t InputCount;

            // Functions

            /**
             * @brief Proces the inputs into an output
             * 
             * @param inputs The inputs to modify, including, as the last value, the bias/threshold
             * @return The output of the neuron
             */
            double ProcessInputs(std::vector<double> &inputs) const;

            /**
             * @brief Stochastic gradient descent method of training a neuron. Stochastic meaning that we work on the live values
             * 
             * @param trainingValues The training values
             * @param learningRate The learning rate, or speed at which weights are modified
             * @return double 
             */
            double TrainNeuron(std::vector<TrainingExample> trainingExamples, double learningRate) noexcept;

        // protected:

            // Properties

            /**
             * @brief A list of weights to apply to an input, including the weight for the bias (which should probably be 1)
             */
            std::vector<double> *m_Weights;

            /**
             * @brief The activation function to apply to the output
             */
            const std::function<double(double)> &m_ActivationFunction;

            // Functions

            /**
             * @brief Produces small random values (-0.05, 0.05) to initalise the weights
             * 
             * @return std::vector<double>* The randomly generate weights, with n (inputCount) + 1 values, where n + 1 is 1.0
             */
            virtual std::vector<double> *GenerateRandomWeights(size_t inputCount);
    };
    
} // End namespace ai_assignment


#endif // H_530093_SRC_NEURON