#include <ctime>
#include <random>
#include <iostream>
#include <functional>

using std::cout, std::endl;

#include "src/Neuron.hpp"
#include "src/activation_functions.hpp"

using namespace ai_assignment;

int main()
{
    // Initalise a 'random' seed for std::rand
    std::srand( std::time(nullptr) );
    
    std::vector<double> weights { 0.5, -0.1, 0.2 };
    std::vector<double> inputs  { 0.0, 1.0, 1.0  };

    Neuron n(2, weights, activation_functions::sigmoidFunc);

    cout << n.ProcessInputs(inputs) << endl;

    std::vector<TrainingExample> ex
    {
        TrainingExample({0.0, 1.0, 1.0}, 1.0)
    };

    for (size_t i = 0; i < 100; i++)
    {
        n.TrainNeuron(ex, 0.05);
    }

    for (size_t i = 0; i < n.m_Weights->size(); i++)
    {
        cout << i << ": " << n.m_Weights->at(i) << endl;
    }

    cout << n.ProcessInputs(inputs) << endl << endl;

    return 0;
}