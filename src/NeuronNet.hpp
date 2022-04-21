#pragma once
#ifndef H_530093_SRC_NET
#define H_530093_SRC_NET 1

#include <vector>

#include "Neuron.hpp"


namespace ai_assignment
{
    /**
     * @brief A network of artifical neurons
     */
    class NeuronNet
    {
        public:

            // Constructors


            NeuronNet();

            /**
             * @brief Copy ctor
             * 
             * @param obj object to copy
             */
            inline NeuronNet(const NeuronNet &obj) noexcept {}

            /**
             * @brief Destroy the NeuronNet object
             */
            inline ~NeuronNet() noexcept {}

            // Functions



        protected:

            // Properties

            /**
             * @brief The architecture of the net
             */
            std::vector<std::vector<Neuron>> m_Architecture;

            /**
             * @brief A 'map' to connect the neurons
             */
            std::vector<double*> m_ConnectionHeuristic;

            /**
             * @brief The number of inputs given
             */
            size_t m_InputCount;
    };
    
} // End namespace ai_assignment


#endif // H_530093_SRC_NET