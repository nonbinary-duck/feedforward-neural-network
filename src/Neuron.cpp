#include "Neuron.hpp"

namespace ai_assignment
{
    // Public constructors

    Neuron::Neuron(size_t inputCount, const std::function<double(double)> &activationFunction) :
        Neuron(inputCount, this->GenerateRandomWeights(inputCount), activationFunction)
    {}

    /**
     * @brief Construct a new 'Neuron' and copy the weights into the heap
     * 
     * @param inputCount The number of inputs to the neuron, excluding the bias/threshold
     * @param weights The starting weights for the 'Neuron'; gets coppied onto the heap and must have an extra 'one' for the bias/threshold
     * @param activationFunction The activation function to apply to the output
     */
    Neuron::Neuron(size_t inputCount, std::vector<double> &weights, const std::function<double(double)> &activationFunction) :
        Neuron(inputCount, new auto(weights), activationFunction)
    {}

    /**
     * @brief Construct a new 'Neuron'
     * 
     * @param inputCount The number of inputs to the neuron, excluding the bias/threshold
     * @param weights The starting weights for the 'Neuron'; must have an extra 'one' for the bias/threshold
     * @param activationFunction The activation function to apply to the output
     */
    Neuron::Neuron(size_t inputCount, std::vector<double> *weights, const std::function<double(double)> &activationFunction) :
        InputCount(inputCount),
        m_ActivationFunction(activationFunction),
        m_Weights(weights)
    {
        // Check that the weights arg is ok
        if (weights->size() == inputCount) throw std::out_of_range("Invalid number of Weights provided, must include an 'extra' weight for the bias/threshold");
        else if (weights->size() != inputCount + 1) throw std::out_of_range("Invalid number of Weights provided");
    }

    // Public Functions

    /**
        * @brief Proces the inputs into an output
        * 
        * @param inputs The inputs to modify, including, as the last value, the bias/threshold
        * @return The output of the neuron
        */
    double Neuron::ProcessInputs(std::vector<double> &inputs) const
    {
        if (inputs.size() != this->InputCount + 1) throw std::invalid_argument("Inputs has incorrect size");

        double output;

        for (size_t i = 0; i < inputs.size(); i++)
        {
            output += inputs[i] * this->m_Weights->at(i);
        }

        return this->m_ActivationFunction(output);
    }

    double Neuron::TrainNeuron(std::vector<TrainingExample> trainingExamples, double learningRate) noexcept
    {
        for (size_t i = 0; i < trainingExamples.size(); i++)
        {
            auto &ex = trainingExamples[i];

            double output = this->ProcessInputs(ex.inputs);

            for (size_t j = 0; j < this->InputCount; i++)
            {
                // wₙ += η(t - o) · xₙ
                this->m_Weights->at(j) += learningRate * (ex.targetOutput - output) * ex.inputs[j];
            }
        }
        
        return 1;
    }

    // Protected Functions

    std::vector<double> *Neuron::GenerateRandomWeights(size_t inputCount)
    {
        auto *weights = new std::vector<double>(inputCount + 1);

        std::random_device rng;
        std::uniform_real_distribution<double> range(-0.05, 0.05);

        for (size_t i = 0; i < inputCount; i++)
        {
            weights->at(i) = range(rng);
        }

        weights->at(inputCount) = 1.0l;
        
        return weights;
    }
}