#include "Neuron.hpp"
#include <concepts>

namespace ai_assignment
{
    // Public constructors


    Neuron::Neuron(size_t inputCount, const activation_func_type activationFunction) :
        Neuron(inputCount, this->GenerateRandomWeights(inputCount), activationFunction)
    {}

    Neuron::Neuron(size_t inputCount, std::vector<double> &weights, const activation_func_type activationFunction) :
        Neuron(inputCount, new auto(weights), activationFunction)
    {}

    Neuron::Neuron(size_t inputCount, std::vector<double> *weights, const activation_func_type activationFunction) :
        InputCount(inputCount),
        m_ActivationFunction(activationFunction),
        m_Weights(weights)
    {
        // Check that the weights arg is ok
        if (weights->size() == inputCount) throw std::out_of_range("Invalid number of Weights provided, must include an 'extra' weight for the bias/threshold");
        else if (weights->size() != inputCount + 1) throw std::out_of_range("Invalid number of Weights provided");
    }


    // Public Functions


    double Neuron::ProcessInputs(std::vector<double*> &inputs) const
    {
        if (inputs.size() != this->InputCount + 1) throw std::invalid_argument("Inputs has incorrect size");

        double output;

        for (size_t i = 0; i < inputs.size(); i++)
        {
            output += (*inputs[i]) * this->m_Weights->at(i);
        }

        return this->m_ActivationFunction(output);
    }

    // Copy-pasted from above, see note in headder file

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

    double Neuron::TrainNeuron(std::vector<TrainingExample> trainingExamples, double learningRate)
    {
        // Setup a value to store the average error rate
        double mseErrorRate = 0.0;
        
        // Loop over the training examples
        for (size_t i = 0; i < trainingExamples.size(); i++)
        {
            // Makes it look neater
            auto &ex = trainingExamples[i];

            // Fetch the result of the inputs
            // (t - o)
            double error = 0;// ex.targetOutput - this->ProcessInputs(ex.inputs);
            mseErrorRate += std::pow(error, 2.0l);

            // Compute "for each linear unit weight wᵢ..."
            for (size_t j = 0; j < this->InputCount + 1; j++)
            {
                // Stochastic gradient descent
                // 
                // wₙ += η(t - o) · xₙ
                // (t - o) == error
                // 
                this->m_Weights->at(j) += learningRate * error * ex.inputs[j];
            }
        }

        return mseErrorRate / trainingExamples.size();
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