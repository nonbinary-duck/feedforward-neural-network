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
        // Mutex
        auto scopeLock = std::lock_guard(this->m_Lock);
        
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
                // Create the neuron with predefined values
                if (startingWeights != nullptr)
                {
                    if (startingWeights->at(i).at(j) == nullptr) throw std::invalid_argument("All elements of the starting weights must be provided");
                    
                    this->m_Architecture.at(i).at(j) = new Neuron(
                        inputs,
                        startingWeights->at(i).at(j),
                        activationFunctions[i]
                    );
                }
                // Use randomly generated starting values
                else this->m_Architecture.at(i).at(j) = new Neuron(inputs, activationFunctions[i]);

                // Create the heuristic for that neuron
                
            }
        }

        // Dispose of the starting weights collection, we don't need the collection any more
        delete startingWeights;
    }
    
} // End namespace ai_assignment