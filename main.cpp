#include <iostream>

using std::cout, std::endl;

#include "src/activation_functions.hpp"

using namespace ai_assignment;

int main()
{
    auto f = activation_functions::sigmoidFunc;

    cout << f(0.9l) << endl;
    cout << f(0.2l) << endl;
    cout << f(0.0l) << endl;
    cout << f(-0.1l) << endl;
    cout << f(-0.9l) << endl;
    
    return 0;
}