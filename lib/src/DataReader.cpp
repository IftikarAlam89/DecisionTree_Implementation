//
// Created by Iftikar sunny on 2019-04-11.
//

#include <thread>
#include "DataReader.hpp"

using boost::algorithm::split;
using boost::timer::cpu_timer;

DataReader::DataReader(const Dataset& dataset) :
    classLabel_(dataset.classLabel),
    trainData_({}),
    testData_({}),
    trainMetaData_({}),
    testMetaData_({}) {
  std::cout << "Start reading data set." << std::endl; cpu_timer timer;
  std::thread readTestingData([this, &dataset]() {
    return processFile(dataset.train.filename, trainData_, trainMetaData_);
  });

  std::thread readTrainingData([this, &dataset]() {
    return processFile(dataset.test.filename, testData_, testMetaData_);
  });

  readTrainingData.join();
  readTestingData.join();
  std::cout << "Done. " << timer.format() << std::endl;

  if (!classLabel_.empty())
    moveClassLabelToBack();

  if (trainData_.empty())
    throw std::runtime_error("Can't open file: " + dataset.train.filename);

  if (testData_.empty())
    throw std::runtime_error("Can't open file: " + dataset.test.filename);
}

void DataReader::processFile(const std::string& filename, Data& data, MetaData &meta) {
  std::ifstream file(filename);
  if (!file)
    return;

  std::string line;
  bool header_loaded = false;

  while (getline(file, line)) {
    if (!header_loaded) {
      parseHeaderLine(line, meta, header_loaded);
    } else {
      parseDataLine(line, data, meta);
    }
  }
  file.close();
}

bool DataReader::parseHeaderLine(const std::string &line, MetaData &meta, bool &header_loaded) {


  if (line.size() == 0) {
    return true;
  }

  if (line[line.find_first_not_of(" ")] == '%') {
    return true;
  }

  if (line.find_first_not_of(" \n\r\t") == line.npos) {
    return true;
  }

  std::string s = line;
  s.erase(0, s.find_first_not_of(" \n\r\t"));
  s.erase(s.find_last_not_of(" \n\r\t") + 1);
  int len = 0;

  len = std::string("@RELATION ").size();
  if (s.size() > (size_t) len
      && strcasecmp(s.substr(0, len).c_str(), "@RELATION ") == 0) {
    return true;
  }

  len = std::string("@ATTRIBUTE ").size();
  if (s.size() > (size_t) len
      && strcasecmp(s.substr(0, len).c_str(), "@ATTRIBUTE ") == 0) {
    s.erase(0, len);
    s.erase(0, s.find_first_not_of(" \n\r\t"));
    len = std::string(" NUMERIC").size();
    if (s.size() > (size_t) len
        && strcasecmp(s.substr(s.size() - len, len).c_str(), " NUMERIC") == 0) {
      s = s.substr(0, s.size() - len);
        meta.labels.push_back(s);
        meta.isClassLabel.push_back(0);
        meta.label_type.push_back(0);
        meta.label_levels.push_back(0);

      return true;
    }

    len = std::string(" REAL").size();
    if (s.size() > (size_t) len
        && strcasecmp(s.substr(s.size() - len, len).c_str(), " REAL") == 0) {
      s = s.substr(0, s.size() - len);
        meta.labels.push_back(s);
        meta.isClassLabel.push_back(0);
        meta.label_type.push_back(0);
        meta.label_levels.push_back(0);
      return true;
    }

      {
          boost::algorithm::trim(s);
          int pos = s.find_last_of("{");
          int pos_l = s.find_last_of("}");

          std::string str_level;
          str_level = s.substr(pos + 1, pos_l - pos - 1);
          // find number of levels in categorical variables
          std::vector<std::string> s_levels;
          split(s_levels, str_level, boost::is_any_of(","));
          size_t levels = s_levels.size();
          s = s.substr(0, pos);
          meta.labels.push_back(s);
          boost::algorithm::trim(s);
          if (strcasecmp(s.c_str(), "CLASS") == 0) {
              meta.isClassLabel.push_back(1);
              for (auto i = s_levels.begin(); i != s_levels.end(); ++i) {
                  meta.Classes.push_back(*i);

              }

          } else
              meta.isClassLabel.push_back(0);


        meta.label_type.push_back(1);
        meta.label_levels.push_back(levels);
      return true;
    }
    return true;
  }

  len = std::string("@DATA").size();
  if (s.size() >= (size_t) len
      && strcasecmp(s.substr(0, len).c_str(), "@DATA") == 0) {

    if (meta.labels.size() > 0) {
      header_loaded = true;
      return true;
    } else {
      return false;
    }
  }

  std::cout << "Symbol not defind " << s.c_str() << std::endl;
  return true;
}

bool DataReader::parseDataLine(const std::string &line, Data &data, MetaData &meta) {
  std::vector<std::string> vec;
  split(vec, line, boost::is_any_of(","));
  trimWhiteSpaces(vec);

  if (classLabel_.empty()) {
    data.emplace_back(std::move(vec));
  } else {
    moveClassDataToBack(vec, meta.labels);
    data.emplace_back(std::move(vec));
  }

  return true;
}

void DataReader::moveClassLabelToBack() {
  const auto result = std::find(std::begin(trainMetaData_.labels), std::end(trainMetaData_.labels), classLabel_);
  if (result != std::end(metaData().labels))
    std::iter_swap(result, std::end(trainMetaData_.labels)-1);
}

void DataReader::moveClassDataToBack(VecS &line, const VecS &labels) const{
  static const auto result = std::find(std::begin(labels), std::end(labels), classLabel_);
  if (result != std::end(labels)) {
    static const auto result_index = std::distance(std::begin(labels), result);
    std::iter_swap(std::begin(line)+result_index, std::end(line)-1);
  }
}

void DataReader::trimWhiteSpaces(VecS &line) {
  for (auto& val: line)
    boost::trim(val);
}
