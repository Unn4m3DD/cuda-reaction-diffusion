#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <iostream>
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#define simulation_width 1024
#define simulation_height 512
#define dA 1
#define dB .7
#define feed_rate .07
#define kill_rate .062
#define time_scale 1
double *current_gen_a;
double *current_gen_b;
double *next_gen_a;
double *next_gen_b;
using namespace std;
__global__ void next(double *current_gen_a, double *current_gen_b,
                     double *next_gen_a, double *next_gen_b) {
  int x = threadIdx.x;
  int y = blockIdx.x;
  if (x == 0 || x == simulation_width - 1 || y == 0 || y == simulation_height)
    return;
  double nabla_a = -current_gen_a[x + simulation_height * y];
  nabla_a += current_gen_a[(x - 1) + simulation_height * y] * 0.2;
  nabla_a += current_gen_a[x + simulation_height * (y - 1)] * 0.2;
  nabla_a += current_gen_a[(x + 1) + simulation_height * y] * 0.2;
  nabla_a += current_gen_a[x + simulation_height * (y + 1)] * 0.2;
  nabla_a += current_gen_a[(x - 1) + simulation_height * (y - 1)] * 0.05;
  nabla_a += current_gen_a[(x - 1) + simulation_height * (y + 1)] * 0.05;
  nabla_a += current_gen_a[(x + 1) + simulation_height * (y - 1)] * 0.05;
  nabla_a += current_gen_a[(x + 1) + simulation_height * (y + 1)] * 0.05;

  double nabla_b = -current_gen_b[x + simulation_height * y];
  nabla_b += current_gen_b[(x - 1) + simulation_height * y] * 0.2;
  nabla_b += current_gen_b[x + simulation_height * (y - 1)] * 0.2;
  nabla_b += current_gen_b[(x + 1) + simulation_height * y] * 0.2;
  nabla_b += current_gen_b[x + simulation_height * (y + 1)] * 0.2;
  nabla_b += current_gen_b[(x - 1) + simulation_height * (y - 1)] * 0.05;
  nabla_b += current_gen_b[(x - 1) + simulation_height * (y + 1)] * 0.05;
  nabla_b += current_gen_b[(x + 1) + simulation_height * (y - 1)] * 0.05;
  nabla_b += current_gen_b[(x + 1) + simulation_height * (y + 1)] * 0.05;
  double a = current_gen_a[x + simulation_height * y];
  double b = current_gen_b[x + simulation_height * y];
  next_gen_a[x + simulation_height * y] =
      a + (dA * nabla_a - a * b * b + feed_rate * (1.0 - a)) * time_scale;
  next_gen_b[x + simulation_height * y] =
      b + (dB * nabla_b + a * b * b - (kill_rate + feed_rate) * b) * time_scale;
}

int main() {
  SDL_Window *window = NULL;
  SDL_Renderer *renderer;
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    fprintf(stderr, "could not initialize sdl2: %s\n", SDL_GetError());
    return 1;
  }
  SDL_CreateWindowAndRenderer(simulation_width, simulation_height, 0, &window,
                              &renderer);
  if (window == NULL) {
    fprintf(stderr, "could not create window: %s\n", SDL_GetError());
    return 1;
  }
  renderer = SDL_GetRenderer(window);
  SDL_UpdateWindowSurface(window);

  cudaMallocManaged(&current_gen_a,
                    simulation_width * simulation_height * sizeof(double));
  cudaMallocManaged(&current_gen_b,
                    simulation_width * simulation_height * sizeof(double));
  cudaMallocManaged(&next_gen_a,
                    simulation_width * simulation_height * sizeof(double));
  cudaMallocManaged(&next_gen_b,
                    simulation_width * simulation_height * sizeof(double));
  for (int i = 0; i < simulation_height * simulation_width; i++) {
    current_gen_a[i] = 1.0;
  }
  cudaMemset(current_gen_b, 0.0,
             simulation_width * simulation_height * sizeof(double));

  for (int z = 0; z < 10; z++) {
    int rand_x = (rand() % (simulation_width - 4));
    int rand_y = (rand() % (simulation_height - 4));
    for (int x = 0; x < 3; x++) {
      for (int y = 0; y < 3; y++) {
        current_gen_a[(rand_x + x) + simulation_height * (rand_y + y)] = 0.0;
        current_gen_b[(rand_x + x) + simulation_height * (rand_y + y)] = 1.0;
      }
    }
  }
  for (int i = 0; i < 100000000; i++) {
    next<<<simulation_height, simulation_width>>>(current_gen_a, current_gen_b,
                                                  next_gen_a, next_gen_b);
    cudaDeviceSynchronize();
    //
    double *tmp = current_gen_a;
    current_gen_a = next_gen_a;
    next_gen_a = tmp;

    tmp = current_gen_b;
    current_gen_b = next_gen_b;
    next_gen_b = tmp;

    if (i % 100 == 0) {
      SDL_RenderClear(renderer);
      for (int y = 0; y < simulation_height; y++) {
        for (int x = 0; x < simulation_width; x++) {
          double current_shade = current_gen_a[x + simulation_height * y] -
                                 current_gen_b[x + simulation_height * y];
          if(current_shade < 0) current_shade = 0;
          if(current_shade > 1) current_shade = 1;
          SDL_SetRenderDrawColor(renderer, (int)(current_shade * 255), (int)(current_shade * 255), (int)(current_shade * 255) ,
                                 255);
          SDL_RenderDrawPoint(renderer, x, y);
          // putpixel(x, y,
          //          (current_shade < .25)   ? WHITE
          //          : (current_shade < .5)  ? LIGHTGRAY
          //          : (current_shade < .75) ? DARKGRAY
          //                                  : BLACK);
        }
      }
      SDL_RenderPresent(renderer);
    }
    SDL_UpdateWindowSurface(window);
  }
  cudaFree(current_gen_a);
  cudaFree(current_gen_b);
  cudaFree(next_gen_a);
  cudaFree(next_gen_b);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
