#include <cassert>
#include <chrono>
#include <cmath>
#include <ctime>
#include <vector>

/// Class to keep the results and stats of each measurement
class PerfResults {
  /// Mean
  double mean = 0.0;
  /// Standard deviation
  double stdev = 0.0;
  /// Start (reset by accumulate)
  std::chrono::high_resolution_clock::time_point start;
  /// Stop (reset by accumulate)
  std::chrono::high_resolution_clock::time_point stop;
  /// Vector with the timings
  std::vector<double> timings;
  /// Locked timings
  bool locked = false;

  /// Generate stats, locks the struct, can't collect stats any more
  void stats() {
    auto size = timings.size();
    // Mean
    double sum = 0.0;
    for (size_t i = 0; i < size; i++)
      sum += timings[i];
    mean = sum / size;
    // Stdev
    sum = 0.0;
    for (size_t i = 0; i < size; i++) {
      double delta = timings[i] - mean;
      sum += delta * delta;
    }
    stdev = std::sqrt(sum / size);
    locked = true;
  }

  /// Return true if time_point hasn't been used yet
  bool isZero(std::chrono::high_resolution_clock::time_point point) {
    return point.time_since_epoch().count() == 0;
  }

  /// Zero a time_point
  void zero(std::chrono::high_resolution_clock::time_point& point) {
    point = std::chrono::high_resolution_clock::time_point();
  }

public:
  /// Starts the timer
  void startTimer() {
    assert(!locked && "Start called after stats produced");
    assert(isZero(start) && "Start called twice");
    start = std::chrono::high_resolution_clock::now();
  }

  /// Stops the timer, accumulates, clears state
  void stopTimer() {
    assert(!locked && "Stop called after stats produced");
    assert(!isZero(start) && "Stop called before start");
    assert(isZero(stop) && "Stop called twice");
    stop = std::chrono::high_resolution_clock::now();
    auto val =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    timings.push_back(val);
    zero(start);
    zero(stop);
  }

  /// Get mean of timings. Locks the timer, only calculate stats once.
  double getMean() {
    if (!locked) {
      assert(isZero(start) && isZero(stop) && "Mismatch call to start/stop");
      stats();
    }
    return mean;
  }

  /// Get stdev of timings. Locks the timer, only calculate stats once.
  double getStdev() {
    if (!locked) {
      assert(isZero(start) && isZero(stop) && "Mismatch call to start/stop");
      stats();
    }
    return stdev;
  }
};
