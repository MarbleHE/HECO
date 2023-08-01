#include "MultiTimer.h"
#include <iostream>
#include <filesystem>
#include <fstream>

MultiTimer::MultiTimer() {
  this->startTimes = new std::vector<TimePoint>;
  this->stopTimes = new std::vector<TimePoint>;
  this->runCounter = 0;
}

MultiTimer::~MultiTimer() {
  delete this->startTimes;
  delete this->stopTimes;
}

int MultiTimer::startTimer() {
  this->runCounter++;
  this->startTimes->reserve(this->startTimes->size() + 1); // Make sure we don't trigger a resize after the timer started
  this->stopTimes->reserve(this->startTimes->size() + 1); // Make sure we don't trigger a resize after the timer started
  int id = (int) this->startTimes->size();
  TimePoint start = std::chrono::high_resolution_clock::now();

  /* We push to both start and stop times.
   * Since we reserved enough places before the push this should, be very quick.
   */
  this->stopTimes->push_back(start);
  this->startTimes->push_back(start);
  return id;
}

void MultiTimer::stopTimer(int id) {
  TimePoint end = std::chrono::high_resolution_clock::now();
  (*this->stopTimes)[id] = end;
  this->runCounter--;
}

TimeDuration MultiTimer::getTime(int id) const {
  if (this->runCounter > 0) {
    // We need to warn that some timer was not stopped
    std::cerr << "[WARN] A timer is still running, while time for timer " << id << " was requested" << std::endl;
  }
  return std::chrono::duration_cast<TimeDuration>((*this->stopTimes)[id] - (*this->startTimes)[id]);
}

void MultiTimer::printToStream(std::ostream& out) {
  bool isFirstInCol = true;
  for (size_t i = 0; i < this->startTimes->size(); ++i) {
    auto t = getTime((int)i).count();
    if (t > 0) {
      if (!isFirstInCol) {
        out << ",";
      }
      else {
        isFirstInCol = false;
      }
      out << t;
    }
    else {
      out << std::endl;
      isFirstInCol = true;
    }
  }
  out << std::endl;
}

void MultiTimer::printToFile(const std::string& path) {
  size_t lastSlashPos = path.find_last_of('/');
  if (lastSlashPos != std::string::npos)
  {
      // Extract the substring from the beginning of the path up to the last '/'
      std::filesystem::create_directories(path.substr(0, lastSlashPos));
  }
  std::ofstream file;
  file.open(path);
  printToStream(file);
  file.close();
}

void MultiTimer::addIteration() {
  TimePoint now = std::chrono::high_resolution_clock::now();
  this->startTimes->push_back(now);
  this->stopTimes->push_back(now);
}
