/*********************************************************
 * Wrapper to create cross-product pairs of benchmarks
 *                Created by Serina Tan 
 *                     Apr 5, 2019 
 * *******************************************************
 * ******************************************************/


// c++ includes
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <functional>
#include <fstream>
#include <thread>
#include <mutex>
#include <cassert>
#include <vector>
#include <functional>
#include <string.h>

// user includes
#include "framework.h"
#include "parboil/benchmarks/interface.h"
#include "cutlass/interface.h"
#include "rodinia/benchmarks/interface.h"
#include "nvidia/interface.h"

std::vector<stream_ops_t> list_stream_ops;
bool synthetic_workload = false;

std::mutex lock_flag;

// called by stream threads
bool set_and_check(int uid, bool start) {
  // this function is guarded by the mutex
  std::lock_guard<std::mutex> guard(lock_flag);

  stream_ops_t& stream_ops = list_stream_ops[uid];
  app_t& app = stream_ops.apps[stream_ops.current_pos];

  if (synthetic_workload) {
    // ignore start sync
    if (start) {
      return true;
    } else {
      if (app.current_pos < app.num_repeat) {
        app.current_pos++;
      }

      if (app.current_pos == app.num_repeat) {
        return true;
      } else {
        return false;
      }
    }
  } else {
    // backward compatibility
    if (start) {
      app.start = true;

      for (auto & s : list_stream_ops) {
        if (!s.apps[s.current_pos].start) return false;
      }

      return true;
    } else {
      app.done = true;

      for (auto & s : list_stream_ops) {
        if (!s.apps[s.current_pos].done) return false;
      }

      return true;
    }
  }
}

// called by stream threads
void set_exit(int uid) {
    std::lock_guard<std::mutex> guard(lock_flag);
    list_stream_ops[uid].exited = true;
}

// called by main thread
bool get_exit(int uid) {
    std::lock_guard<std::mutex> guard(lock_flag);
    return list_stream_ops[uid].exited;
}

// called by stream threads
bool set_and_check_iteration(unsigned uid) {
  std::lock_guard<std::mutex> guard(lock_flag);
  if (synthetic_workload) {
    list_stream_ops[uid].done_iteration = true;

    for (auto & s : list_stream_ops) {
      if (!s.done_iteration) return false;
    }

    return true;
  } else {
    return true;
  }

}


app_t build_app(std::string kernel_arg) {
  app_t result = app_t();

  // split string into argv
  std::vector<std::string> string_argv;
  std::stringstream ss(kernel_arg);
  std::string token;
  while (std::getline(ss, token, ' ')) {
    if (token.length() > 0)
      result.params.push_back(token);
  }

  // select the right benchmark symbol
  if (result.params[0].compare("parb_sgemm") == 0) {
    std::cout << "main: parboil sgemm" << std::endl;
#ifdef PARBOIL_SGEMM
    result.pFunc = main_sgemm;
#endif
  }
  else if (result.params[0].compare( "parb_stencil") == 0) {
    std::cout << "main: parboil stencil" << std::endl;
#ifdef PARBOIL_STENCIL
    result.pFunc = main_stencil;
#endif
  }
  else if (result.params[0].compare( "parb_lbm") == 0) {
    std::cout << "main: parboil lbm" << std::endl;
#ifdef PARBOIL_LBM
    result.pFunc = main_lbm;
#endif
  }
  else if (result.params[0].compare( "parb_spmv") == 0) {
    std::cout << "main: parboil spmv" << std::endl;
#ifdef PARBOIL_SPMV
    result.pFunc = main_spmv;
#endif
  }
  else if (result.params[0].compare( "parb_cutcp") == 0) {
    std::cout << "main: parboil cutcp" << std::endl;
#ifdef PARBOIL_CUTCP
    result.pFunc = main_cutcp;
#endif
  }
  else if (result.params[0].compare( "parb_sad") == 0) {
    std::cout << "main: parboil sad" << std::endl;
#ifdef PARBOIL_SAD
    result.pFunc = main_sad;
#endif
  }
  else if (result.params[0].compare( "parb_histo") == 0) {
    std::cout << "main: parboil histo" << std::endl;
#ifdef PARBOIL_HISTO
    result.pFunc = main_histo;
#endif
  }
  else if (result.params[0].compare( "parb_mriq") == 0) {
    std::cout << "main: parboil mriq" << std::endl;
#ifdef PARBOIL_MRIQ
    result.pFunc = main_mriq;
#endif
  }
  else if (result.params[0].compare( "parb_mrig") == 0) {
    std::cout << "main: parboil mrig" << std::endl;
#ifdef PARBOIL_MRIG
    result.pFunc = main_mrig;
#endif
  }
  else if (result.params[0].compare( "parb_tpacf") == 0) {
    std::cout << "main: parboil tpacf" << std::endl;
#ifdef PARBOIL_TPACF
    result.pFunc = main_tpacf;
#endif
  }
  else if (result.params[0].compare( "parb_bfs") == 0) {
    std::cout << "main: parboil bfs" << std::endl;
#ifdef PARBOIL_BFS
    result.pFunc = main_bfs;
#endif
  }
  else if (result.params[0].compare( "cut_sgemm") == 0) {
    std::cout << "main: cutlass sgemm" << std::endl;
#ifdef CUT_SGEMM
    result.pFunc = main_sgemm;
#endif
  }
  else if (result.params[0].compare( "cut_wmma") == 0) {
    std::cout << "main: cutlass wmma" << std::endl;
#ifdef CUT_WMMA
    result.pFunc = main_wmma;
#endif
  }
  else if (result.params[0].compare( "rod_mummer") == 0) {
    std::cout << "main: rodinia mummer" << std::endl;
#ifdef RODINIA_MUMMER
    result.pFunc = main_mummer;
#endif
  }
  else if (result.params[0].compare( "rod_heartwall") == 0) {
    std::cout << "main: rodinia heartwall" << std::endl;
#ifdef RODINIA_HEARTWALL
    result.pFunc = main_heartwall;
#endif
  }
  else if (result.params[0].compare( "rod_hotspot") == 0) {
    std::cout << "main: rodinia hotspot" << std::endl;
#ifdef RODINIA_HOTSPOT
    result.pFunc = main_hotspot;
#endif
  }
  else if (result.params[0].compare( "rod_cfd") == 0) {
    std::cout << "main: rodinia cfd" << std::endl;
#ifdef RODINIA_CFD
    result.pFunc = main_cfd;
#endif
  }
  else if (result.params[0].compare( "rod_streamcluster") == 0) {
    std::cout << "main: rodinia streamcluster" << std::endl;
#ifdef RODINIA_STREAMCLUSTER
    result.pFunc = main_streamcluster;
#endif
  }
  else if (result.params[0].compare( "rod_pathfinder") == 0) {
    std::cout << "main: rodinia pathfinder" << std::endl;
#ifdef RODINIA_PATHFINDER
    result.pFunc = main_pathfinder;
#endif
  }
  else if (result.params[0].compare( "rod_lavamd") == 0) {
    std::cout << "main: rodinia lavamd" << std::endl;
#ifdef RODINIA_LAVAMD
    result.pFunc = main_lavamd;
#endif
  }
  else if (result.params[0].compare( "rod_myocyte") == 0) {
    std::cout << "main: rodinia myocyte" << std::endl;
#ifdef RODINIA_MYOCYTE
    result.pFunc = main_myocyte;
#endif
  }
  else if (result.params[0].compare( "rod_hotspot3d") == 0) {
    std::cout << "main: rodinia hotspot3d" << std::endl;
#ifdef RODINIA_HOTSPOT3D
    result.pFunc = main_hotspot3d;
#endif
  }

  else if (result.params[0].compare( "nvd_fdtd3d") == 0) {
    std::cout << "main: nvidia fdtd3d" << std::endl;
#ifdef NVD_FDTD3D
    result.pFunc = main_fdtd3d;
#endif
  }
  else if (result.params[0].compare( "nvd_blackscholes") == 0) {
    std::cout << "main: nvidia blackscholes" << std::endl;
#ifdef NVD_BLACKSCHOLES
    result.pFunc = main_blackscholes;
#endif
  }
  else if (result.params[0].compare( "nvd_binomial") == 0) {
    std::cout << "main: nvidia binomial" << std::endl;
#ifdef NVD_BINOMIAL
    result.pFunc = main_binomial;
#endif
  }
  else if (result.params[0].compare( "nvd_sobol") == 0) {
    std::cout << "main: nvidia sobol" << std::endl;
#ifdef NVD_SOBOL
    result.pFunc = main_sobol;
#endif
  }
  else if (result.params[0].compare( "nvd_interval") == 0) {
    std::cout << "main: nvidia interval" << std::endl;
#ifdef NVD_INTERVAL
    result.pFunc = main_interval;
#endif
  }
  else if (result.params[0].compare( "nvd_conv") == 0) {
    std::cout << "main: nvidia conv" << std::endl;
#ifdef NVD_CONV
    result.pFunc = main_conv;
#endif
  }
  else {
    std::cout << "Error: No matching kernels for " <<
              result.params[0] << std::endl;
    abort();
  }

  if (result.pFunc == NULL) {
    std::cout << "Error: Empty function pointer. Check compile defines."
              << std::endl;
    abort();
  }

  return result;
}

void invoke(int uid)
{
  stream_ops_t&stream_ops = list_stream_ops[uid];

  bool can_stop_launching = false;

  while (!can_stop_launching) {
    // Loop over all apps in each iteration
    for (auto & app : stream_ops.apps) {
      int argc = app.params.size();
      char* argv[argc];
      // this vector maintains the original char array pointers
      // cuz the main function will modify the argv
      // this is a sketchy solution
      std::vector<char*> to_free;

      for (int i = 0; i < app.params.size(); i++) {
        argv[i] = new char[app.params[i].length()+1];
        strcpy(argv[i], app.params[i].c_str());
        to_free.push_back(argv[i]);
      }

      std::cout << std::endl;
      std::cout << "******** Invoking " << app.params[0] <<
                " on Stream " << uid << "********" << std::endl;
      std::cout << std::endl;

      // invoke the real function
      app.pFunc(argc, argv, uid, stream_ops.cudaStream);

      // cleanup the char arrays
      for (auto carray: to_free) {
        delete carray;
      }
    }

    can_stop_launching = set_and_check_iteration(uid);
  }

  cudaDeviceSynchronize();
  set_exit(uid);
}


int main(int argc, char** argv) {
  if (argc < 2 || argv[1] == "-h") {
    std::cout << "Usage: ";
    std::cout << "./driver RUNFILE1 [RUNFILE2]" << std::endl;
    abort();
  } 

  for (int i = 1; i < argc; ++i) {
    stream_ops_t stream_ops = stream_ops_t();

    char* filename = argv[i];

    // extract run arguments from file
    // expect a single line file
    std::string line;
    std::ifstream file (filename);
    if (file.is_open())
    {
      while(std::getline (file,line)) {
        if (line.rfind("repeat", 0) == 0) {
          // Repeat previous app
          assert(stream_ops.apps.size() > 0);
          stream_ops.apps.back().num_repeat += 1;
          synthetic_workload = true;
        } else {
          app_t app = build_app(line);
          stream_ops.apps.push_back(app);
        }
      }

      file.close();
    } else {
      std::cout << "Error reading file: " << filename << std::endl;
      abort();
    }

    if (stream_ops.apps.size() > 1) {
      synthetic_workload = true;
    }

    list_stream_ops.push_back(stream_ops);
  }

  for (int i = 0; i < list_stream_ops.size(); i++) {
    std::thread(invoke, i).detach();
  }

  // Sketchy way to join threads without busy wait
  for (unsigned tid = 0; tid < list_stream_ops.size(); tid++) {
    while (!get_exit(tid)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  cudaDeviceSynchronize();

  return 0;
}
