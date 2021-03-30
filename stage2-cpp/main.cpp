#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

#define EQUAL_ERR 0.0001
#define STEADY_STEP 10
#define QOS_LOSS_ERR 0.0001

typedef std::vector<std::vector<float>> Array;

void read_csv(std::string filename, std::vector<std::vector<float>> & result) {
  std::ifstream seq_file(filename);
  if(!seq_file.is_open()) {
    std::cout << filename << " DNE" << std::endl;
    abort();
  }

  std::string line;
  float val;

  while(std::getline(seq_file, line))
  {
    result.push_back(std::vector<float>());
    std::stringstream ss(line);

    // Extract each integer
    while(ss >> val){
      result.back().push_back(val);

      // If the next token is a comma, ignore it and move on
      if(ss.peek() == ',') ss.ignore();
    }
  }
}

float find_qos_loss(const std::vector<float> & scaled_runtime,
                    int num_iter, float seq_sum) {
  // TODO: optimize this
  float scaled_sum = 0;
  for (auto & t : scaled_runtime) {
    scaled_sum += t;
  }

  return (seq_sum * num_iter) / scaled_sum;
}

void simulate(const Array & seq_cycles, const Array & inter1,
    const Array & inter2, int* limits, bool steady=false) {

  float seq_sum[2] = {0, 0};
  for (int i = 0; i < seq_cycles.size(); i++) {
    for (auto & c : seq_cycles[i]) {
      seq_sum[i] += c;
    }
  }

  // keep track of iter
  int iter[2] = {0, 0};

  //scaled_runtimes array is in the form of
  // [[k1 time, k2 time...], [k1 time, k2 time, ...]]
  Array shared_runtimes;
  shared_runtimes.resize(2, std::vector<float>(1, 0));

  // indeces of kernels for two apps - by default 0 and 0
  int kidx[2] = {0, 0};
  // by default the two kernels launch simultaneously
  float remaining_runtimes[2] = {seq_cycles[0][kidx[0]],
                                 seq_cycles[1][kidx[1]]};
  // app that has remaining kernels after the other app finished
  int remaining_app = 0;
  // index of current kernel in remaining app
  int remaining_app_kidx = 0;

  // past and total accumulated runtimes of apps
  float past_qos_loss[2] = {0, 0};
  // list to keep track of estimated qos using steady state estimate
  float steady_state_qos[2] = {-1, -1};
  int steady_state_iter[2] = {0, 0};

  auto handle_completed_kernel = [&](int app_idx) {
    bool can_exit = false;
    int other_app_idx = (app_idx == 0) ? 1 : 0;
    kidx[app_idx] += 1;

    // if app has finished an iteration
    if (kidx[app_idx] == seq_cycles[app_idx].size()) {
      // re-assignment of outer scope variables
      remaining_app = other_app_idx;
      remaining_app_kidx = kidx[other_app_idx];

      kidx[app_idx] = 0;

      // app has completed an iteration of all kernels
      iter[app_idx] += 1;

      // evaluate steady state
      if (iter[app_idx] % STEADY_STEP == 0) {
        // compare qos to the past qos
        float qos_loss = find_qos_loss(shared_runtimes[app_idx],
                                       iter[app_idx],
                                       seq_sum[app_idx]);

        if (std::abs(past_qos_loss[app_idx] - qos_loss) < QOS_LOSS_ERR
            && steady_state_qos[app_idx] == -1) {
          steady_state_qos[app_idx] = qos_loss;
          steady_state_iter[app_idx] = iter[app_idx];

//          std::cout << "check: " << app_idx << ": " << iter[app_idx] << std::endl;
//          std::cout << steady_state_qos[0] << ", " << steady_state_qos[1] << std::endl;

          // Check if we reach steady state for all apps
          can_exit = (steady_state_qos[0] != -1) && (steady_state_qos[1] != -1);
        }

        // update past qos loss to the current qos loss
        past_qos_loss[app_idx] = qos_loss;
      }
    }

    remaining_runtimes[app_idx] = seq_cycles[app_idx][kidx[app_idx]];
    shared_runtimes[app_idx].push_back(0);
    return can_exit;
  };

  // main loop of the simulation
  while (iter[0] < limits[0] and iter[1] < limits[1]) {
    // figure out who finishes first by scaling the runtimes by the
    // slowdowns
    int idx[2] = {kidx[1], kidx[0]};
    float app0_ker_scaled =
                remaining_runtimes[0] / inter1[idx[0]][idx[1]];
    float app1_ker_scaled =
                remaining_runtimes[1] / inter2[idx[0]][idx[1]];

    // Advance scaled runtime by the shorter scaled kernel
    float short_scaled = std::min(app0_ker_scaled, app1_ker_scaled);
    shared_runtimes[0].back() += short_scaled;
    shared_runtimes[1].back() += short_scaled;

    float diff_scaled = std::abs(app0_ker_scaled - app1_ker_scaled);

    bool steady_exit = false;
    if (diff_scaled <= EQUAL_ERR) {
      // both kernels finished at the same time update the total
      // runtime of both kernels to either kernel runtime since they
      // have finished together
      steady_exit |= handle_completed_kernel(0);
      steady_exit |= handle_completed_kernel(1);
    } else if (app0_ker_scaled < app1_ker_scaled) {
      // app0 kernel will finish before app1 kernel update the total
      // runtime of both kernels to app0 kernel runtime since it
      // finished first

      // compute raw remaining runtime of app1
      remaining_runtimes[1] = diff_scaled * inter2[idx[0]][idx[1]];

      steady_exit |= handle_completed_kernel(0);
    } else {
      // app1 kernel will finish before app0 kernel update the total
      // runtime of both kernels to app1 kernel runtime since it
      // finished first

      // compute raw remaining runtime of app0
      remaining_runtimes[0] = diff_scaled * inter1[idx[0]][idx[1]];

      steady_exit |= handle_completed_kernel(1);
    }

    if (steady && steady_exit) {
      break;
    }

  }
  // end of loop

  // finish off the last iteration of remaining app in isolation
  shared_runtimes[remaining_app].back() += remaining_runtimes[remaining_app];
  for (int i = remaining_app_kidx + 1;
       i < seq_cycles[remaining_app].size(); i++) {
    shared_runtimes[remaining_app].push_back(seq_cycles[remaining_app][i]);
  }

  iter[remaining_app] += 1;

  // Handle app that did not get a steady state estimation
  for (int i = 0; i < seq_cycles.size(); i++) {
    if (steady_state_iter[i] == 0) {
      steady_state_iter[i] = iter[i];
      steady_state_qos[i] =
          find_qos_loss(shared_runtimes[i], iter[i], seq_sum[i]);
    }
  }

  if (steady) {
    std::cout << "Stage 2 result: " << steady_state_qos[0] << ", "
              << steady_state_qos[1] << std::endl;
    std::cout << "steady iterations: " << steady_state_iter[0] << ", "
              << steady_state_iter[1] << std::endl;
  }
  else {
//    if not at_least_once:
//  // complete the rest of the required iterations of
//  // remaining app in isolation
//  remaining_iter = self.jobs[remaining_app].num_iters - iter_[
//      remaining_app]
//  isolated_runtime = np.resize(
//      seq_runtimes[remaining_app],
//      remaining_iter * len(seq_runtimes[remaining_app])
//  )
//  shared_runtimes[remaining_app] += list(isolated_runtime)
//
//  // Get rid of tailing zero
//  shared_runtimes = [array[0:-1] if array[-1] == 0 else array
//    for array in shared_runtimes]
//
//  // Build performance instances for full calculation and steady state
//  full_perf = Performance(self.jobs)
//  full_perf.fill_with_duration(shared_runtimes, seq_runtimes,
//                               offset_times=[0, forward_cycles])
//
//  return full_perf
  }

}

int main() {
  // Load input from csv
  std::vector<std::vector<float>> seq_cycles;
  read_csv("seq_cycles.csv", seq_cycles);

  std::vector<std::vector<float>> inter1;
  read_csv("inter1.csv", inter1);

  std::vector<std::vector<float>> inter2;
  read_csv("inter2.csv", inter2);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  int limits[2] = {2400, 2800};

  bool steady = true;
  simulate(seq_cycles, inter1, inter2, limits, steady);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Time difference = " <<
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
            << "[us]" << std::endl;

  return 0;
}
