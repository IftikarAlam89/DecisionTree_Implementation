
//
// Created by Iftikar sunny on 2019-04-11.
//

#include <cmath>
#include <algorithm>
#include <iterator>
#include "Calculations.hpp"
#include "Utils.hpp"
# include<random>
#include <omp.h>

using std::tuple;
using std::pair;
using std::forward_as_tuple;
using std::vector;
using std::string;
using std::unordered_map;

bool Calculations::sortbysec(const std::pair<std::string,int> &a,
                             const std::pair<std::string,int> &b)
{
    return (a.second < b.second);
}
const bool Calculations::isNumeric(std::string value) {
    if (bool empty = value.empty(); !empty) {
        try {
            std::stod(value);
        } catch (const std::exception &e) {
            return false;
        }
    }
    return true;
}

bool Calculations::sortnorm(const std::pair<std::string,int> &a,
                             const std::pair<std::string,int> &b)
{   
    return std::stod(a.first)<std::stod(b.first);
}



tuple<const DataObject, const DataObject> Calculations::partition(DataObject& data, const Question& q) {
    DataObject true_rows;
   DataObject false_rows;


    true_rows.features.resize(data.features.size());
    //std::cout<<data.features.size()<<'\n';
   false_rows.features.resize(data.features.size());
    //for( auto i=data.features[0].begin();i!=data.features[0].end();++i){
        //std::cout<<(*i).first<<' '<<(*i).second<<'\n';
    //}
    //std::for_each(data.features.begin(), data.features.end(), [](auto &n) { std::sort(n.begin(), n.end(),sortbysec); });
    //for( auto i=data.features[1].begin();i!=data.features[1].end();++i){
        //std::cout<<(*i).first<<' '<<(*i).second<<'\n';
    //}
    int col = q.column_;
    double num_features = data.features.size();
    
   // #pragma  omp parallel for
    for (auto j = data.features[col].begin(); j <data.features[col].end(); ++j) {
        //#pragma omp task
        if (q.solve((*j))){
             true_rows.index_class[(*j).second]=data.index_class[(*j).second];
             
             int len = std::distance(data.features[col].begin(), j);
            
            for (int i = 0; i <= num_features-1; ++i) {
                
                //std::cout<<"Number of features "<<num_features<<" row number  "<<len<<'\n';
                
               // true_rows.features[i].push_back(data.features[i][len]);
                true_rows.features[i].push_back(std::move(data.features[i][len]));
                

            }
            }
        else{
            int len = std::distance(data.features[col].begin(), j);
           false_rows.index_class[(*j).second]=data.index_class[(*j).second];
             
           for (int i = 0; i <= num_features-1; ++i) {
               // int len = std::distance(data.features[col].begin(), j);
                //std::cout<<"Number of features "<<num_features<<" row number  "<<len<<'\n';
                false_rows.features[i].push_back(std::move(data.features[i][len]));
            }
            }
    }
    //#pragma omp taskwait
    //#pragma  omp parallel for
    //for (auto k = true_rows.features[0].begin(); k <true_rows.features[0].end(); ++k){
       // true_rows.index_class[(*k).second]=data.index_class[(*k).second];
    //}
    //#pragma  omp parallel for
    //for (auto k = false_rows.features[0].begin(); k < false_rows.features[0].end(); ++k){
       // false_rows.index_class[(*k).second]=data.index_class[(*k).second];
    //}
    //true_rows.index_class=data.index_class;
    //false_rows.index_class=data.index_class;

   // std::cout<<"Split at "<<q.column_<<" with value  "<<q.value_<<'\n';

    //std::cout<<"Number of features in left split "<<true_rows.features.size()<<" row number  "<<true_rows.features[0].size()<<' '<<true_rows.index_class.size()<<'\n';
    //std::cout<<"Number of features in right split "<<false_rows.features.size()<<" row number  "<<false_rows.features[0].size()<<' '<<false_rows.index_class.size()<<'\n';
    //std::for_each(true_rows.features.begin(), true_rows.features.end(), [](auto &n) { std::sort(n.begin(), n.end()); });
    //std::for_each(false_rows.features.begin(), false_rows.features.end(), [](auto &n) { std::sort(n.begin(), n.end()); });
    //for (int i=0;i<5;++i){
        //std::cout<<true_rows.features[0][i].first<<' '<<true_rows.features[7][i].first<<' '<<true_rows.features[10][i].first<<'\n';
    //}

    return forward_as_tuple(true_rows, false_rows);

}

tuple<const double, const Question> Calculations::find_best_split(DataObject& data,  MetaData& meta,const ClassCounter& counts) {
    double best_gain = 0.0;
    auto best_question = Question();
    double N=data.features[0].size();

    double gini_tot=gini(counts,N);

    int num_features=data.features.size();


       #pragma omp parallel for
         
         
         for (int i = 0; i < num_features; ++i) {
            
            tuple<std::string, double> best_pair;

            best_pair = determine_best_threshold_numeric(data, i, counts, meta);

            double gain = gini_tot - std::get<1>(best_pair);

            #pragma omp critical
             
   	    if (gain>best_gain){
                best_gain=gain;
                best_question = Question(i, std::get<0>(best_pair));
                }		

    }


    if (meta.label_type[best_question.column_]){
        meta.number_used[best_question.column_]+=1;

    }

    return forward_as_tuple(best_gain, best_question);
}





const double Calculations::gini(const ClassCounter& counts, double N) {
    double impurity = 1.0;
    for (auto i=counts.begin();i!=counts.end();++i){
        impurity -=pow((*i).second/N,2);
    }

    //TODO: compute gini index, given class counts and the dataset size
    return impurity;
}


const double Calculations::diffgini(const ClassCounter& counts, double N,const ClassCounter& newCounts) {
    //std::cout<<"      "<<counts.size()<<'\n';
    double impurity = 1.0;
    auto j=counts.begin();

    for (auto k=newCounts.begin();k!=newCounts.end();++k){
        int diff;
        //std::cout<<(*k).first<<' '<<(*j).first<<'\n';
        //std::cout<<(*j).second<<' '<<(*k).second<< ' '<< N<<'\n';
        diff=(*j).second-(*k).second;
        ++j;

        //std::cout<<diff<<'\n';
        impurity -=pow(diff/N,2);

    }

    //TODO: compute gini index, given class counts and the dataset size
    return impurity;
}

tuple<std::string, double> Calculations::determine_best_threshold_numeric(DataObject& data, int col,const ClassCounter& counts,const MetaData &meta) {
    double best_loss = std::numeric_limits<float>::infinity();
    std::string best_thresh;
    if(meta.label_type[col] && (meta.label_levels[col]-meta.number_used[dist])<=1){
        continue;
    }
    ClassCounter newCount=Calculations::resetclassCounts(meta);
    auto k=data.features[col].begin();
    if(meta.label_type[col]){

    std::sort(data.features[col].begin(), data.features[col].end());
    }else{
    std::sort(data.features[col].begin(), data.features[col].end(),sortnorm);
    }
    int N=data.features[col].size();
    int num_feat=data.features.size();
    //std::cout<<num_feat<<'\n';
    for (auto i=data.features[col].begin();i!=data.features[col].end()-1;++i){
        int dist=std::distance(data.features[col].begin(),i);
        newCount[data.index_class.at((*i).second)]+=1;
        auto j=i+1;
        if((*i).first != (*j).first){
            //std::cout<<(*i).first<<' '<<(*j).first<<'\n';


            float p =(float)(dist+1)/(N);
            //std::cout<<" Len is "<< len<<" N is "<<N<<" p is "<<p<<'\n';
            double gini_left=gini(newCount,dist+1);
            //std::cout<<" Gini for left is "<< gini_left<<'\n';
            double gini_right=diffgini(counts,N-dist-1,newCount);
            //std::cout<<" Gini for right is "<< gini_right<<'\n';
            double loss=(p*gini_left+((1-p)*gini_right));
            //std::cout<<" p is "<< (len/N)<<'\n';
            //std::cout<<" Loss is "<< loss<<'\n';
            //int lenth=std::distance(data.features[col].begin(), k);
            //k=j;
            //int lenth2=std::distance(data.features[col].begin(), j);
            //std::cout<<" Len before "<< lenth<<" lenth after "<<lenth2<<'\n';
            if(loss<best_loss){
                best_loss=loss;
                best_thresh=(*i).first;
            }
        }
    }
    std::sort(data.features[col].begin(), data.features[col].end(),sortbysec);
    //std::cout<<" left gain is "<< gini_left1<<'\n';
    //std::cout<<" right gain is "<< gini_right2<<'\n';
    //TODO: find the best split value for a discrete ordinal feature
    return forward_as_tuple(best_thresh, best_loss);
}


tuple<std::string, double> Calculations::determine_best_threshold_cat(const DataObject& data, int col,const ClassCounter& counts,const MetaData &meta) {
    double best_loss = std::numeric_limits<float>::infinity();
    std::string best_thresh;
    ClassCounter newCount=resetclassCounts(meta);
    auto k=data.features[col].begin();
    int N=data.features[col].size();

    for (auto i=data.features[col].begin();i!=data.features[col].end()-1;++i){
        auto j=i+1;
        if((*i).first != (*j).first){
            //std::cout<<(*i).first<<'\n';
            for (k;k!=j;++k){
                int dist=std::distance(data.features[col].begin(),k);
                std::cout<<dist<<' '<<data.index_class.at(dist)<<' '<<N<<'\n';
                //newCount[data.index_class.at(dist)]+=1;
                //std::cout<<newCount[data.index_class.at(dist)]<<'\n';
            }
            double len=std::distance(data.features[col].begin(), i);
            double gini_left=gini(newCount, len);
            double gini_right=diffgini(counts,N-len,newCount);
            double loss=(((float)len/N)*gini_left+((1-(float)len/N)*gini_right));
            if(loss<best_loss) {
                best_loss = loss;
                best_thresh = (*i).first;
            }
            k=j;

        }
    }
    return forward_as_tuple(best_thresh, best_loss);
}

const ClassCounter Calculations::resetclassCounts(const MetaData &meta){
    ClassCounter counter;
    for (auto i=meta.Classes.begin();i!=meta.Classes.end();++i){
        counter[*i]=0;
    }
    return counter;
}


const ClassCounter Calculations::classCounts(const DataObject& data,const MetaData &meta) {
    ClassCounter counter;
    for (auto i=meta.Classes.begin();i!=meta.Classes.end();++i){
        counter[*i]=0;
    }
    for (auto rows=data.features[0].begin();rows!=data.features[0].end();++rows) {
       // counter[(*rows).second] += 1;
       counter[data.index_class.at((*rows).second)] += 1;

    }
    return counter;
}

const DataObject Calculations::bootstrap(const DataObject& data){
    DataObject bootstrap;
    bootstrap.features.resize(data.features.size());
    int N=data.features[0].size();
    int num_features=data.features.size();
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(0, N-1);
    for(int i=0;i<N;++i){
        int k=distr(eng);
        for (int j=0;j<num_features;++j){
            bootstrap.features[j].push_back(data.features[j][i]);

        }
    }
    return bootstrap;


}

DataObject Calculations::createData(const Data& data,std::mt19937_64 ran) {
    DataObject dt;
    int len=data[0].size();
    //std::cout<<len<<'\n';
    dt.features.resize(len-1);
    std::cout<<dt.features.size()<<'\n';
    for(auto i =data.begin();i!=data.end();++i){

        int dist=std::distance(data.begin(),i);
        //std::cout<<dist<<'\n';
        for(int j=0;j<len-1;++j){
            //std::cout<<(*i)[j]<<'\n';
            std::pair<std::string, int> l=std::make_pair((*i)[j],dist);
            dt.features[j].emplace_back(l);
            //std::cout<<j<<'\n';
        }
        //std::cout<<dist<<' '<<(*i).back()<<'\n';
        dt.index_class[dist]=(*i)[len-1];
        //std::cout<<dt.index_class[dist]<<' '<<(*i).back()<<'\n';
    }
    //std::for_each(dt.features.begin(), dt.features.end(), [](auto &n) { std::sort(n.begin(), n.end()); });

    return dt;

}

const Data Calculations::bootstrap_dr(const Data& data,std::mt19937_64 ran){
    Data bootstrap;
   // bootstrap.resize(data.size());
    int N=data.size();
   //std::cout<<data.size()<<' '<<bootstrap.size()<<'\n';
   // std::random_device rd; // obtain a random number from hardware
   ran.s
    std::uniform_int_distribution<> distr(0, N-1);
    for(int i=0;i<N;++i){
        int k=distr(ran);

            bootstrap.push_back(data[k]);


    }
    //std::cout<<data.size()<<' '<<bootstrap.size()<<'\n';
    return bootstrap;


}

const std::vector<int> Calculations::bootstrap_dr(const Data& data,std::mt19937_64 ran){

    // bootstrap.resize(data.size());
    int N=data.size();
    //std::cout<<data.size()<<' '<<bootstrap.size()<<'\n';
    // std::random_device rd; // obtain a random number from hardware

    std::uniform_int_distribution<> distr(0, N-1);
    for(int i=0;i<N;++i){
        int k=distr(ran);

        bootstrap.push_back(data[k]);


    }
    //std::cout<<data.size()<<' '<<bootstrap.size()<<'\n';
    return bootstrap;


}
