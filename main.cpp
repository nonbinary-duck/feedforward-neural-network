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

int main()
{
    // Initalise a 'random' seed for std::rand
    std::srand( std::time(nullptr) );
    
    auto *startingWeights = new vector< vector < vector< double >* > >
    (
        {
            { // Layer 0
                new vector< double >({ // Neuron 0
                    0.5,
                    -0.2,
                    0.5
                }),
                new vector< double >({ // Neuron 1
                    0.1,
                    0.2,
                    0.3
                })
            },
            { // Layer 1
                new vector< double >({ // Neuron 0
                    0.7,
                    0.6,
                    0.2
                }),
                new vector< double >({ // Neuron 1
                    0.9,
                    0.8,
                    0.4
                })
            }
        }
    );
    
    NeuralNet *n = new NeuralNet(
        vector<size_t>{2, 2},
        3,
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
                .inputs = { 0.0, 1.0, 1.0 },
                .targetOutput = { 1.0, 1.0 }
            }
        }
    );

    auto out = n->ProcessInputs({0.0, 1.0, 1.0});

    for (size_t i = 0; i < out->size(); i++)
    {
        cout << out->at(i) << endl;
    }
    
    delete out;

    n->TrainNetwork(ex, 0.05);

    return 0;
}