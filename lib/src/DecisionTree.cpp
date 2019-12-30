

#include "DecisionTree.hpp"
#include <future>

using std::make_shared;
using std::shared_ptr;
using std::string;
using boost::timer::cpu_timer;

DecisionTree::DecisionTree(const DataReader& dr) : root_(Node()), dr_(dr) {
  std::cout << "Start building tree." << std::endl; cpu_timer timer;
  DataObject dt=createDataObject(dr_.trainData());

  root_ = buildTree(dt, dr_.metaData());
  std::cout << "Done. " << timer.format() << std::endl;
}

DecisionTree::DecisionTree(const DataReader& dr,std::mt19937_64 ran) : root_(Node()), dr_(dr) {
    std::cout << "Start building tree." << std::endl; cpu_timer timer;
    DataObject dt=createDataObjectBag(dr_.trainData(),ran);
    root_ = buildTree(dt, dr_.metaData());
    std::cout << "Done. " << timer.format() << std::endl;
}

MetaData DecisionTree::createMetaObject(const MetaData& data) {
    MetaData dt;
    int len=data.labels.size();
    //std::cout<<len<<'\n';
    dt.labels.resize(len);
    dt.number_used.resize(len);
    dt.label_levels.resize(len);
    dt.label_type.resize(len);
    dt.isClassLabel.resize(len);
    dt.Classes=data.Classes;


    for(int i =0;i<len;++i) {
        dt.labels.push_back(data.labels[i]);
        std::cout<<data.label_levels[i]<<' '<<data.label_type[i]<<'\n';
        dt.number_used.push_back(0);
        dt.label_levels.push_back(data.label_levels[i]);
        dt.label_type.push_back(data.label_type[i]);
        dt.isClassLabel.push_back(data.isClassLabel[i]);;

    }

    return dt;

}

DataObject DecisionTree::createDataObject(const Data& data) {
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

DataObject DecisionTree::createDataObjectBag(const Data& data,std::mt19937_64 ran) {
    DataObject dt;
    int len=data[0].size();
    //std::cout<<len<<'\n';
    dt.features.resize(len-1);
    int N =data.size();
    std::random_device rd;
    ran(rd());
    std::uniform_int_distribution<unsigned long long>> distr(0, N-1);
    //std::cout<<dt.features.size()<<'\n';
    for(auto i =data.begin();i!=data.end();++i){

        int dist=distr(ran);
        int dist1=std::distance(data.begin(),i);
        //std::cout<<dist<<'\n';
        for(int j=0;j<len-1;++j){
            //std::cout<<(*i)[j]<<'\n';
            std::pair<std::string, int> l=std::make_pair(data[dist][j],dist1);
            dt.features[j].emplace_back(l);
            //std::cout<<j<<'\n';
        }
        //std::cout<<dist<<' '<<(*i).back()<<'\n';
        dt.index_class[dist1]=data[dist][len-1];
        //std::cout<<dt.index_class[dist]<<' '<<(*i).back()<<'\n';
    }
    //std::for_each(dt.features.begin(), dt.features.end(), [](auto &n) { std::sort(n.begin(), n.end()); });

    return dt;

}


const Node DecisionTree::buildTree(DataObject& data, MetaData& meta) {
    //std::cout<<meta.Classes.size()<<'\n';
    //std::cout<<"decision tree started"<<'\n';
    ClassCounter counts =Calculations::classCounts(data,meta);
    std::tuple<const double, const Question> split= Calculations::find_best_split(data, meta,counts);
    double gain=std::get<0> (split);
    //std::cout<<gain<<'\n';
    Question  q=std::get<1> (split);

    Node n1;
    if (gain < 0.001||data.features[0].size()<10){
        Leaf l1=Leaf(counts);
        n1=Node(l1);
        //std::cout<<"node started"<<'\n';

    }else {
        std::tuple<const DataObject, const DataObject> data_partition =Calculations::partition(data, q);
        //std::cout<<"here"<<'\n';
        DataObject left_data=std::get<0> (data_partition);

        DataObject right_data=std::get<1> (data_partition);
        //if (left_data.features.empty()||right_data.features.empty()){
           // Leaf l1=Leaf(counts);
            //n1=Node(l1);
        //} else {
          
           //#pragma omp parallel
            //{
           auto l=std::async(std::launch::async,&DecisionTree::buildTree,this,std::ref(left_data),std::ref(meta));
           const Node n_left=l.get();
           //std::cout<<"here now"<<'\n';
           auto r=std::async(std::launch::async,&DecisionTree::buildTree,this,std::ref(right_data),std::ref(meta));
           const Node n_right=r.get();
           // Node n_left=buildTree(left_data,meta);
            //Node n_right=buildTree(right_data,meta);
            //}
            //#pragma omp single
            n1=Node(n_left,n_right,q);
       // 

    }
  return n1;
}

void DecisionTree::print() const {
  print(make_shared<Node>(root_));
}

void DecisionTree::print(const shared_ptr<Node> root, string spacing) const {
  if (bool is_leaf = root->leaf() != nullptr; is_leaf) {
    const auto &leaf = root->leaf();
    std::cout << spacing + "Predict: "; Utils::print::print_map(leaf->predictions());
    return;
  }
  std::cout << spacing << root->question().toString(dr_.metaData().labels) << "\n";

  std::cout << spacing << "--> True: " << "\n";
  print(root->trueBranch(), spacing + "   ");

  std::cout << spacing << "--> False: " << "\n";
  print(root->falseBranch(), spacing + "   ");
}

void DecisionTree::test() const {
  TreeTest t(dr_.testData(), dr_.metaData(), root_);

}
