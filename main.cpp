#include <ctime>
#include <random>
#include <iostream>
#include <functional>

using std::cout, std::endl, std::vector;

#include "src/utils.hpp"
#include "src/Neuron.hpp"
#include "src/activation_functions.hpp"

using namespace ai_assignment;

int main()
{
    vector<double*> f = 
    {
        new double(1.0),
        new double(2.3),
        new double(0.0)
    };

    for (size_t i = 0; i < f.size(); i++)
    {
        cout << i << ": " << *f[i] << endl;
    }

    endl(cout);
    
    delete f[2];

    f[2] = f[0];

    for (size_t i = 0; i < f.size(); i++)
    {
        cout << i << ": " << *f[i] << endl;
    }

    endl(cout);

    *f[2] = 2.0;

    for (size_t i = 0; i < f.size(); i++)
    {
        cout << i << ": " << *f[i] << endl;
    }

    return 0;
    
    // Initalise a 'random' seed for std::rand
    std::srand( std::time(nullptr) );
    
    vector<double> weights { 0.0, 0.0, 1.0 };
    vector<vector<double>> inputs
    {
        { 0.0, 0.0, 1.0  },
        { 0.0, 1.0, 1.0  },
        { 1.0, 0.0, 1.0  },
        { 1.0, 1.0, 1.0  }
    };

    Neuron n(2, weights, activation_functions::sigmoidFunc);

    vector<TrainingExample> ex
    {
        TrainingExample({0.0, 0.0, 1.0}, 0.0),
        TrainingExample({0.0, 1.0, 1.0}, 1.0),
        TrainingExample({1.0, 0.0, 1.0}, 1.0),
        TrainingExample({1.0, 1.0, 1.0}, 0.0)
    };

    long double previous;
    long double current;
    size_t it = 0;

    for (size_t i = 0; i < 10000; i++)
    {
        current = n.TrainNeuron(ex, 0.05);
        
        if (it == 0 && previous == current) {it = i;}
        
        cout << "er: " << current << " Δ" << previous - current << ((it != 0)? " :: mse reached equilibrium at " + std::to_string(it) : "") << endl;

        previous = current;
    }

    for (size_t i = 0; i < inputs.size(); i++)
    {
        cout << "Input " << i << ": " << n.ProcessInputs(inputs[i]) << endl;
    }

    utils::printWeights(&n);

    return 0;
}