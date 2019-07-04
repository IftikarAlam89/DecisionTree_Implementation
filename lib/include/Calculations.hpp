/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#ifndef DECISIONTREE_CALCULATIONS_HPP
#define DECISIONTREE_CALCULATIONS_HPP

#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include <boost/timer/timer.hpp>
#include "Question.hpp"
#include "Utils.hpp"
#include <random>

using ClassCounter = std::unordered_map<std::string, int>;

namespace Calculations {

    std::tuple<const DataObject, const DataObject> partition(DataObject &data, const Question &q);

    const double gini(const ClassCounter& counts, double N);

    const double diffgini(const ClassCounter& counts, double N,const ClassCounter& newCounts);


    std::tuple<const double, const Question> find_best_split(DataObject &rows,  MetaData &meta,const ClassCounter& counts);

    std::tuple<std::string, double> determine_best_threshold_numeric(DataObject &data, int col,const ClassCounter& counts,const MetaData &meta);

    std::tuple<std::string, double> determine_best_threshold_cat(const DataObject &data, int col,const ClassCounter& counts,const MetaData &meta);
    bool sortbysec(const std::pair<std::string,int> &a,
                   const std::pair<std::string,int> &b);

    const ClassCounter classCounts(const DataObject &data, const MetaData &meta);

    const ClassCounter resetclassCounts(const MetaData &meta);
    const DataObject bootstrap(const DataObject& data) ;
    DataObject createData(const Data& data);
    const Data bootstrap_dr(const Data& data,std::mt19937_64 ran);
   bool sortnorm(const std::pair<std::string,int> &a,
                                  const std::pair<std::string,int> &b);
    const bool isNumeric(std::string value) ;

} // namespace Calculations

#endif //DECISIONTREE_CALCULATIONS_HPP
