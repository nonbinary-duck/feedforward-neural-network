#include "NeuralNet.hpp"


namespace ai_assignment
{
    // Public Constructors
    
    NeuralNet::NeuralNet(
                const vector<size_t> netArchitecture,
                const size_t inputs,
                const vector< activation_func_type > activationFunctions,
                vector< vector < vector< double >* > > *startingWeights
            )
        : m_NetArchitecture(netArchitecture), m_Inputs(inputs)
    {
        this->InitialiseNeurons(netArchitecture, inputs, activationFunctions, startingWeights);

        // Dispose of the starting weights collection, we don't need the collection any more
        delete startingWeights;
    }

    NeuralNet::NeuralNet(const NeuralNet &obj) noexcept
        : m_Architecture(obj.m_Architecture),
            m_NetArchitecture(obj.m_NetArchitecture),
            m_Inputs(obj.m_Inputs)
    {
        // Create new values on the heap
        
        // Create a new value on the heap for each neuron
        for (size_t i = 0; i < this->m_Architecture.size(); i++)
        {
            for (size_t j = 0; j < this->m_Architecture.at(i).size(); j++)
            {
                this->m_Architecture.at(i).at(j) = new auto(*this->m_Architecture.at(i).at(j));
            }
        }
    }


    // Public Functions

    
    vector<double> *NeuralNet::ProcessInputs(vector<double> inputs, vector<vector<double>> *recordedOutputs = nullptr)
    {
        // Lock the mutex and release on destruction (return)
        auto scopedLock = std::scoped_lock(this->m_Lock);
        
        // Check the input is valid
        size_t inputCount = inputs.size();
        
        if (inputCount != this->m_Inputs) throw std::invalid_argument("Input provided doesn't match architecture");
        
        // Execute the neurons layer-by-layer
        size_t layerCount = this->m_NetArchitecture.size();

        // Create a copy of the inputs to store outputs in
        // The copy is neccicary so that we keep the very last value (bias/threshold)
        vector<double> outputs = vector<double>(inputs);
        vector<double> *finalOutputs = new vector<double>(
            this->m_NetArchitecture.at(layerCount - 1) - 1
        );

        for (size_t i = 0; i < layerCount; i++)
        {
            for (size_t j = 0; j < this->m_NetArchitecture.at(i); j++)
            {
                // Use the output of the previous layer to input into each neuron on this layer

                // If this is the last layer, fill out the final outputs
                if (i + 1 == layerCount)
                {
                    finalOutputs->at(j) = this->m_Architecture.at(i).at(j)->ProcessInputs(inputs);
                }
                // Otherwise, fill in the outputs for the next layer
                else
                {
                    outputs.at(j) = this->m_Architecture.at(i).at(j)->ProcessInputs(inputs);
                }
            }

            // Copy the outputs of this layer to use as the inputs of the next layer
            inputs = outputs;

            // Record the outputs if it wants us to
            if (recordedOutputs != nullptr)
            {
                if (i + 1 == layerCount) recordedOutputs->at(i) = outputs;
            }
        }

        return finalOutputs;
    }

    double NeuralNet::TrainNetwork(vector<TrainingExample> &trainingExamples)
    {
        for
    }


    // Protected Functions


    void NeuralNet::InitialiseNeurons(
        const vector<size_t> &netArchitecture,
        const size_t inputs,
        const vector< activation_func_type > &activationFunctions,
        vector< vector < vector< double >* > > *startingWeights
    )
    {
        // Initalise the architecture to the correct size
        this->m_Architecture = std::vector<std::vector<Neuron*>>(netArchitecture.size());
        
        // Create the neurons for each layer
        for (size_t i = 0; i < netArchitecture.size(); i++)
        {
            // Initalise this layer of the architecture
            this->m_Architecture.at(i) = std::vector<Neuron*>(netArchitecture[i]);

            // Setup the neurons in this layer
            // There cannot be more than inputs than values in this ANN, since each neuron has exactly the same number of inputs. This is something which could be easily changed in the futire.
            for (size_t j = 0; j < netArchitecture[i]; j++)
            {
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
    }

    
} // End namespace ai_assignment