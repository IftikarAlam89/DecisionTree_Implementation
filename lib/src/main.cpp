//
// Created by Iftikar sunny on 2019-04-11.
//

#include <iostream>
#include "DataReader.hpp"
#include "DecisionTree.hpp"
#include "Bagging.hpp"

int main() {
    Dataset d;
    d.train.filename = "/Users/iftikarsunny/data_assignment_3/covtype.arff";
    d.test.filename = "/Users/iftikarsunny/data_assignment_3/covtype_test.arff";

    DataReader dr(d);
    DecisionTree dt(d);
    //dt.print();
    //dt.test();
    Bagging bc(d, 5);
    bc.test();
    //std::string s="{Apple,Grape,Lemon,Lime,Eggplant,Beets,Pepper,Radish}";
    //std::cout<<s<<'\n';

    return 0;
}
