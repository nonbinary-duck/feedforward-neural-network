#pragma once
#ifndef H_530093_SRC_CONNECTION
#define H_530093_SRC_CONNECTION 1

#include <vector>

#include "Neuron.hpp"


namespace ai_assignment
{
    /**
     * @brief A single coordinate
     */
    struct Coordinate
    {
        size_t i, j;
    };
    
    
    /**
     * @brief A data type to hold information on how two things connect
     */
    class Connection
    {
        public:
            // Constructors & Destructors
            
            inline Connection(long inputLabel, Neuron *processor, Connection *nextLink)
                : m_InputLabel(inputLabel),
                    m_OutputLabel(-1),
                    ProcessorNeuron(processor)
            {}

            
            inline Connection(const Connection &obj) noexcept
                : ProcessorNeuron(obj.ProcessorNeuron),
                    m_InputLabel(obj.m_InputLabel),
                    m_OutputLabel(obj.m_OutputLabel)
            {}
            
            /**
             * @brief Destroy our processor
             */
            inline ~Connection() noexcept
            {
                delete this->ProcessorNeuron;
            }

            // Properties
            
            /**
             * @brief A neuron, which we own, that processes this value
             */
            Neuron *ProcessorNeuron;

            /**
             * @brief The next link in the chain
             */
            Connection *NextLink;

            // Accessors

            inline long GetInputLabel() const noexcept { return this->m_InputLabel; }
            inline long GetOutputLabel() const noexcept { return this->m_OutputLabel; }


        protected:

            /**
             * @brief The label for the input, -1 if not an input
             */
            long m_InputLabel;

            /**
             * @brief The label for the output, -1 if not an output
             */
            long m_OutputLabel;
    };

} // End namespace ai_assignment


#endif // H_530093_SRC_CONNECTION