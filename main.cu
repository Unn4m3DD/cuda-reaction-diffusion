#include <malloc.h>
#include <string.h>

#include <iostream>
double *current_gen_a;
double *current_gen_b;
double *next_gen_a;
double *next_gen_b;
int simulation_width = 256;
int simulation_height = 256;
double dA = 1;
double dB = .5;
double feed_rate = .055;
double kill_rate = .062;
double time_scale = 1;
using namespace std;
__global__ void next(double *current_gen_a, double *current_gen_b,
                     double *next_gen_a, double *next_gen_b,
                     int simulation_width, int simulation_height, double dA,
                     double dB, double feed_rate, double kill_rate,
                     double time_scale) {
  int x = threadIdx.x;
  int y = blockDim.x;
  double nabla_a = -current_gen_a[x * simulation_height + y];
  nabla_a += current_gen_a[x - 1 * simulation_height + y] * 0.2;
  nabla_a += current_gen_a[x * simulation_height + y - 1] * 0.2;
  nabla_a += current_gen_a[x + 1 * simulation_height + y] * 0.2;
  nabla_a += current_gen_a[x * simulation_height + y + 1] * 0.2;
  nabla_a += current_gen_a[x - 1 * simulation_height + y - 1] * 0.05;
  nabla_a += current_gen_a[x - 1 * simulation_height + y + 1] * 0.05;
  nabla_a += current_gen_a[x + 1 * simulation_height + y - 1] * 0.05;
  nabla_a += current_gen_a[x + 1 * simulation_height + y + 1] * 0.05;

  double nabla_b = -current_gen_b[x * simulation_height + y];
  nabla_b += current_gen_b[x - 1 * simulation_height + y] * 0.2;
  nabla_b += current_gen_b[x * simulation_height + y - 1] * 0.2;
  nabla_b += current_gen_b[x + 1 * simulation_height + y] * 0.2;
  nabla_b += current_gen_b[x * simulation_height + y + 1] * 0.2;
  nabla_b += current_gen_b[x - 1 * simulation_height + y - 1] * 0.05;
  nabla_b += current_gen_b[x - 1 * simulation_height + y + 1] * 0.05;
  nabla_b += current_gen_b[x + 1 * simulation_height + y - 1] * 0.05;
  nabla_b += current_gen_b[x + 1 * simulation_height + y + 1] * 0.05;

  double a = current_gen_a[x * simulation_height + y];
  double b = current_gen_b[x * simulation_height + y];
  next_gen_a[x * simulation_height + y] =
      a + (dA * nabla_a - a * b * b + feed_rate * (1.0 - a)) * time_scale;
  next_gen_b[x * simulation_height + y] =
      b + (dB * nabla_b + a * b * b - (kill_rate + feed_rate) * b) * time_scale;
}

int main() {
  cudaMallocManaged(&current_gen_a,
                    simulation_width * simulation_height * sizeof(double));
  cudaMallocManaged(&current_gen_b,
                    simulation_width * simulation_height * sizeof(double));
  cudaMallocManaged(&next_gen_a,
                    simulation_width * simulation_height * sizeof(double));
  cudaMallocManaged(&next_gen_b,
                    simulation_width * simulation_height * sizeof(double));

  cudaMemset(current_gen_a, 1,
             simulation_width * simulation_height * sizeof(double));
  cudaMemset(current_gen_b, 0,
             simulation_width * simulation_height * sizeof(double));

  for (int z = 0; z < 15; z++) {
    int rand_x = (rand() % (simulation_width - 6)) + 3;
    int rand_y = (rand() % (simulation_height - 6)) + 3;
    for (int x = 0; x < 3; x++) {
      for (int y = 0; y < 3; y++) {
        current_gen_a[(rand_x + x) * simulation_height + rand_y + y] = 0.0;
        current_gen_b[(rand_x + x) * simulation_height + rand_y + y] = 1.0;
      }
    }
  }
  cout << "yhey" <<endl;
  for (int i = 0; i < 10000; i++) {
    next<<<256, 256>>>(current_gen_a, current_gen_b, next_gen_a, next_gen_b,
                       simulation_width, simulation_height, dA, dB, feed_rate,
                       kill_rate, time_scale);
    cudaDeviceSynchronize();

    double *tmp = current_gen_a;
    current_gen_a = next_gen_a;
    next_gen_a = tmp;

    tmp = current_gen_b;
    current_gen_b = next_gen_b;
    next_gen_b = tmp;
  }

  for (int x = 0; x < simulation_width; x++) {
    for (int y = 0; y < simulation_width; y++) {
      double current_shade = current_gen_a[x * simulation_height + y] -
                             current_gen_b[x * simulation_height + y];
      cout << (current_shade < .2) ? " "
      : (current_shade < .4)       ? "░"
      : (current_shade < .6)       ? "▒"
      : (current_shade < .8)       ? "▓"
                                   : "█";
    }
    cout << endl;
  }
  cudaFree(current_gen_a);
  cudaFree(current_gen_b);
  cudaFree(next_gen_a);
  cudaFree(next_gen_b);
}
