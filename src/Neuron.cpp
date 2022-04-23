#include "Neuron.hpp"

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

    double Neuron::TrainNeuron(std::vector<Example> &trainingExamples, double learningRate)
    {
        // Setup a value to store the average error rate
        double mseErrorRate = 0.0;
        
        // Loop over the training examples and adjust the mean squared error rate
        for (size_t i = 0; i < trainingExamples.size(); i++)
        {
            double error = (
                        // Fetch the error term of the inputs
                        // (t - o)
                        trainingExamples[i].targetOutput -
                        this->ProcessInputs(trainingExamples[i].inputs)
            );
            mseErrorRate += std::pow(
                error,
                2.0
            );

            this->TrainNeuron(trainingExamples[i].inputs, error,learningRate);
        }

        return mseErrorRate / trainingExamples.size();
    }

    void Neuron::TrainNeuron(std::vector<double> &inputs, double error, double &learningRate)
    {
        // Compute "for each linear unit weight wᵢ..."
        for (size_t j = 0; j < this->InputCount + 1; j++)
        {
            // Stochastic gradient descent
            // 
            // wₙ += η(t - o) · xₙ
            // (t - o) == error
            // 
            this->m_Weights->at(j) += learningRate * error * inputs[j];
        }
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