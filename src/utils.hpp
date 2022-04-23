#pragma once
#ifndef H_530093_SRC_UTILS
#define H_530093_SRC_UTILS 1

#include <vector>
#include <iostream>


namespace ai_assignment::utils
{
    /**
     * @brief Releases the values inside a vector from the heap. This does nothing to the vector itself
     * 
     * @tparam T The base type of the vector (not a pointer type)
     * @param vec The vector which stores the pointers
     */
    template<typename T>
    inline void releaseVecValues(std::vector<T*> &vec)
    {
        // Iterate over the pointers and remove them
        for (size_t i = 0; i < vec.size(); i++)
        {
            if (vec[i] == nullptr) continue;
            
            delete vec[i];

            // Prevent double free
            vec[i] = nullptr;
        }
    }

    /**
     * @brief Releases the values inside a vector from the heap. This does nothing to the vector itself
     * 
     * @tparam T The base type of the vector (not a pointer type)
     * @param vec The vector which stores the pointers
     */
    template<typename T>
    inline void releaseVecValues(std::vector<std::vector<T*>> &vec)
    {
        // Iterate over the pointers and remove them
        for (size_t i = 0; i < vec.size(); i++)
        {
            releaseVecValues(vec[i]);
        }
    }

} // End namespace utils

#endif // H_530093_SRC_UTILS