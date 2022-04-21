#include "NeuralNet.hpp"


namespace ai_assignment
{
    NeuralNet::NeuralNet(
                vector<size_t> netArchitecture,
                const vector<size_t> inputArchitecture,
                const size_t inputs,
                const vector< activation_func_type > activationFunctions,
                vector< vector < vector< double >* > > *startingWeights
            )
        : m_InputArchitecture(inputArchitecture)
    {
        // Initalise the heuristics to the correct size (the number of layers of ANN + the input layer)
        this->m_ConnectionHeuristic = vector<vector<double*>>(inputArchitecture.size() + 1);

        // Create the first layer of the heuristics...

        this->m_ConnectionHeuristic.at(0) = vector<double*>(inputs);        

        // ...and assign some values on the heap for it
        for (size_t i = 0; i < inputs; i++)
        {
            this->m_ConnectionHeuristic.at(0).at(i) = new double(0.0);
        }
        

        // Initalise the architecture to the correct size
        this->m_Architecture = std::vector<std::vector<Neuron*>>(netArchitecture.size());
        
        // Create the neurons for each layer
        for (size_t i = 0; i < netArchitecture.size(); i++)
        {
            // Initalise this layer of the architecture
            this->m_Architecture.at(i) = std::vector<Neuron*>(netArchitecture[i]);

            // Initalise this layer of the heuristics
            // Connection heuristics are offset by -1 because of the input layer
            this->m_ConnectionHeuristic.at(i + 1) = vector<double*>(
                // The heuristic will either have as many values as there are inputs for the next layer, or it will have the sum of the neural networks (only if it's on the last layer)
                (i != netArchitecture.size() - 1)?
                    inputs
                    :
                    netArchitecture[i]
            );

            // Setup the neurons in this layer
            // There cannot be more than inputs than values in this ANN, since each neuron has exactly the same number of inputs. This is something which could be easily changed in the futire.
            for (size_t j = 0; j < inputs; j++)
            {
                // Create the heuristic for either this neuron or the input

                // If this is a neuron or if it's an input, create a new value on the heap
                // (neuron || input) == (nCount + iCount) > j
                if ( !( netArchitecture[i] + inputArchitecture[i] > j ) )
                {
                    this->m_ConnectionHeuristic.at(i + 1).at(j) = new double();
                }
                // If it's not an input or a neuron, reference the last value of the previous layer (a.k.a. threshold/bias)
                else
                {
                    // This will never fail, since zero is always assigned and we start from 1
                    this->m_ConnectionHeuristic.at(i + 1).at(j)
                        = this->m_ConnectionHeuristic.at(i).at(inputs);
                }

                // If there isn't a neuron here, skip to the next value
                if (!(netArchitecture[i] < j)) continue;
                
                // Create the neuron with predefined values
                if (startingWeights != nullptr)
                {
                    // Gives a descript error of what went wrong
                    if (startingWeights->at(i).at(j) == nullptr) throw std::invalid_argument("All elements of the starting weights must be provided");
                    
                    // Crete a new neuron using the provided weights
                    this->m_Architecture.at(i).at(j) = new Neuron(
                        inputs,
                        startingWeights->at(i).at(j),
                        activationFunctions[i]
                    );
                }
                // Use randomly generated starting values
                else this->m_Architecture.at(i).at(j) = new Neuron(inputs, activationFunctions[i]);
            }
        }

        // Dispose of the starting weights collection, we don't need the collection any more
        delete startingWeights;
    }
    
} // End namespace ai_assignment