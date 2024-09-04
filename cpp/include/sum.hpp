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

template<typename real_iterator>
typename std::iterator_traits<real_iterator>::value_type
recursive_sum(real_iterator begin, real_iterator end)
{
    typedef typename std::iterator_traits<real_iterator>::value_type real;
    /*
     * This function performs recursive pairwise summation of a vector of
     * real numbers.
     * The resulting error should be in the order of
     *        O(eps * sqrt(log(n))) to O(eps * log(n))
     * (https://en.wikipedia.org/wiki/Pairwise_summation), slightly worse than
     * Kahan summation but better than simple linear aggregation.
     */
    const std::ptrdiff_t size = end - begin;
    if (size < 0)
        throw std::runtime_error(
            "Negative size: pointers/iterators are corrupt in recursive_sum."
        );
    if (size == 0)
        return real();
    if (size == 1)
        return *begin;

    /* Create a temporary array that is always filled continuously from the
     * front with the results of the previous pairwise summation.
     * Hence, we can always use step width 1 and only need to adjust the
     * end iterator.
     */
    std::vector<real> tmp((size / 2) + (size % 2));
    auto out = tmp.begin();

    /* When we first start this routine, nothing has been written to tmp
     * yet. So, the first iteration is special in that it pulls from the
     * input iterator rather than from the previous output:
     */
    auto it_in = begin;
    for (; it_in < end; ++it_in){
        if (out == tmp.end()){
            std::cerr << "ERROR out == end!!!\n";
            return tmp[0];
        }
        /* Set the output variable to the first summand: */
        *out = *it_in;

        /* Go for the second summand, if it exists: */
        ++it_in;

        /* If a second summand is in the vector, add it to the output
         * variable: */
        if (it_in != end)
            *out += *it_in;

        /* Go to the next output position: */
        ++out;
    }

    /* The end of the current output is the next iteration's input end: */
    auto out_end = out;

    /* Restart the input and output from/to the beginning of the temporary
     * vector: */
    auto it = tmp.cbegin();
    out = tmp.begin();
    while (out_end - it > 1)
    {
        for (; it < out_end; ++it){
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
            if (it != out_end)
                *out += *it;

            /* Go to the next output position: */
            ++out;
        }

        /* The end of the current output is the next iteration's input end: */
        out_end = out;

        /* Restart the input and output from/to the beginning of the temporary
         * vector: */
        it = tmp.cbegin();
        out = tmp.begin();
    }

    return tmp.front();
}

template<typename real>
real recursive_sum(const std::vector<real>& x)
{
    return recursive_sum(x.cbegin(), x.cend());
}

}

#endif