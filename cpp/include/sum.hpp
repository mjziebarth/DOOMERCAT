/*
 * Templates for summation algorithms with reduced roundoff error.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2024 Technische Universität München
 *
 * Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
 * the European Commission - subsequent versions of the EUPL (the "Licence");
 * You may not use this work except in compliance with the Licence.
 * You may obtain a copy of the Licence at:
 *
 * https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 */

#ifndef DOOMERCAT_SUM_HPP
#define DOOMERCAT_SUM_HPP

#include <vector>
#include <limits>
#include <iostream>

namespace doomercat {

template<typename real>
real recursive_sum(const std::vector<real>& x)
{
    /*
     * This function performs recursive pairwise summation of a vector of
     * real numbers.
     * The resulting error should be in the order of
     *        O(eps * sqrt(log(n))) to O(eps * log(n))
     * (https://en.wikipedia.org/wiki/Pairwise_summation), slightly worse than
     * Kahan summation but better than simple linear aggregation.
     */
    if (x.empty())
        return real();
    if (x.size() == 1)
        return x.front();

    /* Create a temporary array that is always filled continuously from the
     * front with the results of the previous pairwise summation.
     * Hence, we can always use step width 1 and only need to adjust the
     * end iterator.
     */
    std::vector<real> tmp((x.size() / 2) + (x.size() % 2));
    auto out = tmp.begin();

    /* When we first start this routine, nothing has been written to tmp
     * yet. A neat way to handle the start is to use the constant iterators
     * of the input vector here:
     */
    auto it = x.cbegin();
    auto end = x.cend();
    while (end - it > 1)
    {
        for (; it < end; ++it){
            if (out == tmp.end()){
                std::cerr << "ERROR out == end!!!\n";
                return tmp[0];
            }
            /* Set the output variable to the first summand: */
            *out = *it;

            /* Go for the second summand, if it exists: */
            ++it;

            /* If a second summand is in the vector, add it to the output
             * variable: */
            if (it != end)
                *out += *it;

            /* Go to the next output position: */
            ++out;
        }

        /* The end of the current output is the next iteration's start: */
        end = out;

        /* Restart the input and output from/to the beginning of the temporary
         * vector: */
        it = tmp.cbegin();
        out = tmp.begin();
    }

    return tmp.front();
}

}

#endif