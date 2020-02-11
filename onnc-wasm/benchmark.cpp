extern "C" {
  #include "benchmark.h"
}

#include <cstdint>
#include <unordered_map>
#include <string>
#include <cstdio>
#include <sys/time.h>

namespace host {
class Timer {
public:
  Timer() = default;
  virtual ~Timer() = default;

  virtual void clear() {
    IsRecording = false;
    RecTime = 0;
  }

  void start() {
    gettimeofday(&TStart, nullptr);
    IsRecording = true;
  }

  uint64_t stop() {
    if (IsRecording) {
      struct timeval TEnd;
      gettimeofday(&TEnd, nullptr);
      RecTime += (uint64_t)1000000 * (TEnd.tv_sec - TStart.tv_sec) +
                 TEnd.tv_usec - TStart.tv_usec;
      IsRecording = false;
    }
    return RecTime;
  }

private:
  bool IsRecording = false;
  struct timeval TStart;
  uint64_t RecTime = 0;
};

static Timer& getTimer(const std::string& key){
    static std::unordered_map<std::string, Timer> timers;
    return timers[key];
}
}
void host_QITC_time_start(char* key){
    host::getTimer(std::string(key)).start();
};
void host_QITC_time_stop(char* key, char* message){
    fprintf(stderr, "[STATS] %s cost %ld us\n", message, host::getTimer(std::string(key)).stop());
};
void host_QITC_time_clear(char* key){
    host::getTimer(std::string(key)).clear();
};
