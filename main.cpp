#include <ctime>
#include <random>
#include <iostream>
#include <functional>

using std::cout, std::endl, std::vector;

#include "src/utils.hpp"
#include "src/Neuron.hpp"
#include "src/NeuralNet.hpp"
#include "src/activation_functions.hpp"

using namespace ai_assignment;

namespace ai_assignment::utils
{
    inline void printWeights(NeuralNet *n)
    {
        // Fetch a copy
        auto *weights = n->GetWeights();
        
        for (size_t i = 0; i < weights->size(); i++)
        {
            for (size_t j = 0; j < weights->at(i).size(); j++)
            {
                for (size_t k = 0; k < weights->at(i).at(j).size(); k++)
                {
                    std::cout << "layer: " << i << " neuron: " << j << " weight: " << k << ": " << weights->at(i).at(j).at(k) << std::endl;
                }
            }
        }
        
        // Dispose
        delete weights;
    }
} // End namespace utils


int main()
{
    // Initalise a 'random' seed for std::rand
    std::srand( std::time(nullptr) );
    
    auto *startingWeights = new vector< vector < vector< double >* > >
    (
        {
            { // Layer 0
                new vector< double >({ // Neuron a4
                    0.74,
                    0.8,
                    0.35,
                    0.9
                }),
                new vector< double >({ // Neuron a5
                    0.13,
                    0.4,
                    0.97,
                    0.45,
                }),
                new vector< double >({ // Neuron a6
                    0.68,
                    0.1,
                    0.96,
                    0.36
                })
            },
            { // Layer 1
                new vector< double >({ // Neuron a7
                    0.35,
                    0.5,
                    0.9,
                    0.98
                }),
                new vector< double >({ // Neuron a8
                    0.8,
                    0.13,
                    0.8,
                    0.92
                })
            }
        }
    );
    
    NeuralNet *n = new NeuralNet(
        vector<size_t>{3, 2},
        4,
        vector<Neuron::activation_func_type>
        {
            activation_functions::sigmoidFunc,
            activation_functions::noFunc
        },
        startingWeights
    );

    auto ex = vector< NeuralNet::Example >
    (
        {
            {
                .inputs = { 0.5, 1.0, 0.75, 1.0 },
                .targetOutput = { 1.0, 0.0 }
            },
            {
                .inputs = { 1.0, 0.5, 0.75, 1.0 },
                .targetOutput = { 1.0, 0.0 }
            },
            {
                .inputs = { 1.0, 1.0, 1.0, 1.0 },
                .targetOutput = { 1.0, 0.0 }
            },
            {
                .inputs = { -0.01, 0.5, 0.25, 1.0 },
                .targetOutput = { 0.0, 1.0 }
            },
            {
                .inputs = { 0.5, -0.25, 0.13, 1.0 },
                .targetOutput = { 0.0, 1.0 }
            },
            {
                .inputs = { 0.01, 0.02, 0.05, 1.0 },
                .targetOutput = { 0.0, 1.0 }
            }
        }
    );

    cout << "Network trained for " << n->TrainNetwork(ex, 0.1) << " epochs" << endl << endl;

    utils::printWeights(n);

    auto *out = n->ProcessInputs({0.3, 0.7, 0.9, 1.0});

    cout << endl << "Outputs of unseen input:" << endl;

    for (size_t i = 0; i < out->size(); i++)
    {
        cout << 'y' << i + 1 << ": " << out->at(i) << endl;
    }

    delete out;

    return 0;
}