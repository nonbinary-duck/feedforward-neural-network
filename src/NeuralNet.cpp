#include "NeuralNet.hpp"


namespace ai_assignment
{
    // Public Constructors
    
    NeuralNet::NeuralNet(
                const vector<size_t> netArchitecture,
                const size_t inputs,
                const vector< Neuron::activation_func_type > activationFunctions,
                vector< vector < vector< double >* > > *startingWeights
            )
        : m_NetArchitecture(netArchitecture), m_Inputs(inputs)
    {
        this->InitialiseNeurons(netArchitecture, inputs, activationFunctions, startingWeights);

        // Dispose of the starting weights collection, we don't need the collection any more
        if (startingWeights != nullptr) delete startingWeights;
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

    
    vector<double> *NeuralNet::ProcessInputs(vector<double> inputs, vector<vector<double>> *recordedOutputs)
    {
        // Lock the mutex and release on destruction (return)
        auto scopedLock = std::scoped_lock(this->m_Lock);
        
        // Check the input is valid
        size_t inputCount = inputs.size();
        
        if (inputCount != this->m_Inputs) throw std::invalid_argument("Input provided doesn't match architecture");
        
        // Create a copy of the inputs to store outputs in
        // The copy is neccicary so that we keep the very last value (bias/threshold)
        size_t layerCount = this->m_NetArchitecture.size();

        vector<double> outputs = vector<double>(inputs);
        vector<double> *finalOutputs = new vector<double>(
            this->m_NetArchitecture.at(layerCount - 1)
        );

        // Execute the neurons layer-by-layer
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
            if (recordedOutputs != nullptr) recordedOutputs->at(i) = outputs;
        }

        return finalOutputs;
    }

    size_t NeuralNet::TrainNetwork(vector<Example> &trainingExamples, double learningRate)
    {
        // Acquire lock
        auto scopedLock = std::scoped_lock(this->m_Lock);

        // Setthe first mean squared error to an arbitrarily large value to avoid thinking that it's getting worse on the first iteration
        double mse = 1E300;
        double previousMSE;
        size_t epochs = 0;

        // Setup the storage for the results
        auto *outputCache = new vector<vector<double>>(this->m_NetArchitecture.size());
        // Cache the current weights to go back if we're not improving the situation
        auto *prevWeights = this->GetWeights();
        // And run the same method to give us a properly initialised object to store new weights in
        auto *newWeights = this->GetWeights();
        
        // Places to write to for the assignment
        // File writing code snippet taken from https://en.cppreference.com/w/cpp/io/manip/setprecision
        std::fstream errCsv;
        errCsv.open("err.csv");
        std::fstream weightsCsv;
        weightsCsv.open("weights.csv");

        weightsCsv << std::setprecision(std::numeric_limits<double>::digits10 + 1);
        errCsv << std::setprecision(std::numeric_limits<double>::digits10 + 1);

        // Loop until break
        while (true)
        {
            // Print the weights to the output file (very slow)
            this->PrintWeights(weightsCsv);
            
            // Copy the last mse to be the previous one
            previousMSE = mse;
            mse = 0.0;
            epochs++;
            
            for (size_t i = 0; i < outputCache->size(); i++)
            {
                outputCache->at(i) = vector<double>(this->m_Inputs);
            }
            
            for (size_t i = 0; i < trainingExamples.size(); i++)
            {
                mse += this->TrainNetwork(trainingExamples[i], learningRate, outputCache, newWeights);
            }

            mse /= trainingExamples.size();

            errCsv << epochs << ',' << mse << std::endl;

            // If this epoch has made things worse, revert that epoch and end the training
            if (mse > previousMSE)
            {
                // Log the weights from this epoch
                this->PrintWeights(weightsCsv);
                // Update to the previous weights
                this->SetWeights(prevWeights);
                // Log the final weights
                // But label that data
                weightsCsv << "# Revert update ↓" << std::endl;
                this->PrintWeights(weightsCsv);
                // Exit
                break;
            }

            // Break when epoch 135 has finished, as discussed in the assignment.
            if (epochs == 135) break;

            // Otherwise, continue in a loop until the mean squared error stops changing
            if (mse == previousMSE) break;

            // Copy the new weights into the previous weights so we can revert to them if the training went badly
            *prevWeights = *newWeights;
        }

        // Cleanup
        errCsv.close();
        weightsCsv.close();
        delete outputCache;
        delete prevWeights;
        delete newWeights;

        return epochs;
    }

    double NeuralNet::TrainNetwork(Example &trainingExample, double &learningRate, vector<vector<double>> *sharedOutputCache, weight_type *newWeights)
    {
        // Propagate the input forward through the network
        auto out = this->ProcessInputs(trainingExample.inputs, sharedOutputCache);

        // Create a place to store error terms for the neurons
        // Include the hidden error terms
        auto errorTerms = vector<vector<double>>(this->m_NetArchitecture.size());

        // Get the mean variance from the example to return
        double returnErr = 0.0;

        // Initalise it for the outputs
        errorTerms[this->m_NetArchitecture.size() - 1] = vector<double>(out->size());

        for (size_t k = 0; k < out->size(); k++)
        {
            // T4.3
            // errorTerms[this->m_Architecture.size() - 1][k] =
            // (
            //     (
            //         out->at(k) *
            //         (
            //             1.0 -
            //             out->at(k)
            //         )
            //     ) *
            //     (
            //         trainingExample.targetOutput[k] - out->at(k)
            //     )
            // );

            errorTerms[this->m_Architecture.size() - 1][k] = trainingExample.targetOutput[k] - out->at(k);

            // (t - o)²
            // Squared error
            returnErr += std::pow(trainingExample.targetOutput[k] - out->at(k), 2);
        }

        // We're done with the output, free it from the heap
        delete out;

        // Loop over the neurons back to front to "backpropigate"
        // Needs to be signed otherwise it'll underflow to 2⁶⁴ - 1
        // Exclude the output layer, which was already accounted for
        for (long i = this->m_NetArchitecture.size() - 2; i >= 0; i--)
        {
            // Initalise it to the correct size
            errorTerms[i] = vector<double>(this->m_NetArchitecture[i]);
            
            // Loop over each unit in this layer
            for (size_t j = 0; j < this->m_NetArchitecture[i]; j++)
            {
                // T4.4
                double sumErr = 0.0;

                // Σ weight of node in front ⨉ error of node in front
                // Loop over the neurons in front of us and multiply their error term with the weight assigned to the input they take from us
                for (size_t k = 0; k < this->m_NetArchitecture[i + 1]; k++)
                {
                    // j is the input they take from us, i is the layer ahead and k is the node in that layer ahead
                    sumErr += this->m_Architecture[i + 1][k]->m_Weights->at(j)
                        // We then get the error term of that node
                        * errorTerms[i + 1][k];
                }
                
                // Cache the result of the output of this neuron
                // Slower, but looks neater
                double o = sharedOutputCache->at(i).at(j);

                errorTerms[i][j] = o * (1.0 - o) * sumErr;
            }
        }

        // Then update the network weights
        // Every layer
        for (size_t i = 0; i < this->m_NetArchitecture.size(); i++)
        {
            // Every neuron
            for (size_t j = 0; j < this->m_NetArchitecture[i]; j++)
            {
                // The weights in that neuron
                // this->m_Inputs == this->m_Architecture[i][j]->m_Weights->size()
                for (size_t k = 0; k < this->m_Inputs; k++)
                {
                    // T4.5
                    // To get Δw we need the inputs to this neuron, which could be from another neuron or the example
                    this->m_Architecture[i][j]->m_Weights->at(k) +=
                    (
                        learningRate * (
                            errorTerms[i][j] *
                            // The input to this weight of the neuron
                            (
                                (i == 0)?
                                    trainingExample.inputs.at(k)
                                    :
                                    sharedOutputCache->at(i - 1).at(k)
                            )
                        )
                    );

                    // Give the caller the new weights
                    newWeights->at(i).at(j).at(k) = this->m_Architecture[i][j]->m_Weights->at(k);
                }
                
            }
        }
        
        // Return the figure generated earlier as the square error (t - o)
        // We squared it earlier, which also means we have the absolute value
        return returnErr;
    }


    // Protected Functions


    void NeuralNet::InitialiseNeurons(
        const vector<size_t> &netArchitecture,
        const size_t inputs,
        const vector< Neuron::activation_func_type > &activationFunctions,
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
                        inputs - 1,
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