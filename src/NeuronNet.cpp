#include "NeuronNet.hpp"


namespace ai_assignment
{
    NeuronNet::NeuronNet(
                vector<size_t> netArchitecture,
                const vector<size_t> inputArchitecture,
                const size_t inputs,
                const vector< activation_func_type > activationFunctions,
                vector< vector < vector< double >* > > *startingWeights
            )
        : m_InputArchitecture(inputArchitecture)
    {
        // Initalise the architecture to the correct size
        this->m_Architecture = std::vector<std::vector<Neuron*>>(netArchitecture.size());
        
        // Create the neurons for each layer
        for (size_t i = 0; i < netArchitecture.size(); i++)
        {
            // Initalise this layer of the architecture
            this->m_Architecture.at(i) = std::vector<Neuron*>(netArchitecture[i]);

            // Setup the neurons in this layer
            for (size_t j = 0; j < netArchitecture[i]; j++)
            {
                if (startingWeights != nullptr)
                {
                    this->m_Architecture.at(i).at(j) = new Neuron(
                        inputs,
                        startingWeights->at(i).at(j),
                        activationFunctions[i]
                    );
                }
                else this->m_Architecture.at(i).at(j) = new Neuron(inputs, activationFunctions[i]);
            }
        }
        
    }
    
} // End namespace ai_assignment