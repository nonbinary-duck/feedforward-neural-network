#pragma once
#ifndef H_530093_SRC_NEURON
#define H_530093_SRC_NEURON 1

#include <functional>
#include <stdexcept>
#include <random>
#include <vector>

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
            inline Neuron(size_t inputCount, const std::function<double(double)> &activationFunction) noexcept :
                Neuron(inputCount, )

            /**
             * @brief Construct a new 'Neuron'
             * 
             * @param inputCount The number of inputs to the neuron, excluding the bias/threshold
             * @param weights The starting weights for the 'Neuron'; gets coppied onto the heap and must have an extra 'one' for the bias/threshold
             * @param activationFunction The activation function to apply to the output
             */
            inline Neuron(size_t inputCount, std::vector<double> &weights, const std::function<double(double)> &activationFunction) noexcept :
                InputCount(inputCount),
                m_ActivationFunction(activationFunction),
                m_Weights(new auto(weights))
            {
                // Check that the weights arg is ok
                if (weights.size() == inputCount) throw std::out_of_range("Invalid number of Weights provided, must include an 'extra' weight for the bias/threshold");
                else if (weights.size() != inputCount + 1) throw std::out_of_range("Invalid number of Weights provided");
            }

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
            inline double ProcessInputs(std::vector<double> &inputs) const
            {
                if (inputs.size() != this->InputCount + 1) throw std::invalid_argument("Inputs has incorrect size");

                double output;

                for (size_t i = 0; i < inputs.size(); i++)
                {
                    output += inputs[i] * this->m_Weights->at(i);
                }

                return this->m_ActivationFunction(output);
            }

        protected:

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
             * @return std::vector<double>* 
             */
            virtual std::vector<double> *GenerateRandomWeights(size_t weightCount)
            {
                auto *weights = new std::vector<double>(weightCount + 1);

                auto a = std::uniform_real_distribution<double>();

                for (size_t i = 0; i < weightCount; i++)
                {
                    weights[i] = a() std::rand();
                }
                
            }
    };
    
} // End namespace ai_assignment


#endif // H_530093_SRC_NEURON