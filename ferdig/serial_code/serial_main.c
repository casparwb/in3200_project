#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "functions.h"


int main(int argc, char *argv[]){
  int m, n, c, iters;
  float kappa;
  image u, u_bar;
  unsigned char *image_chars, *new_chars;
  char *input_jpeg_filename, *output_jpeg_filename;

  clock_t  start, end;


  input_jpeg_filename = argv[1];
  output_jpeg_filename = argv[2];

  kappa = atof(argv[3]);
  iters = atoi(argv[4]);

  import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
  u.m = m; u.n = n;
  u_bar.m = m; u_bar.n = n;
  printf("Import successful! m = %d | n = %d \n", m, n);

  new_chars = (unsigned char*)malloc(m*n*sizeof(unsigned char));

  allocate_image(&u, m, n);
  allocate_image(&u_bar, m, n);
  printf("Allocation successful!\n");

  convert_jpeg_to_image(image_chars, &u);
  printf("Convertion to image successful!\n");

  start = clock();
  iso_diffusion_denoising(&u, &u_bar, kappa, iters);
  printf("Denoising successful!\n");
  end = clock();

  convert_image_to_jpeg(&u_bar, new_chars);
  printf("Convertion to jpeg successful!\n");

  export_JPEG_file(output_jpeg_filename, new_chars, m, n, c, 75);

  deallocate_image(&u);
  deallocate_image(&u_bar);
  printf("Deallocation successful!\n");

  printf("Serial iso denoising time: %lds,  %ldms\n", ((end - start))/CLOCKS_PER_SEC, ((end - start)*1000)/CLOCKS_PER_SEC);

  return 0;
}
