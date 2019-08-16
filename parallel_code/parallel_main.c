#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
//#include "functions.h"



typedef struct{
  float **image_data;  /* a 2D array of floats */
  int m;               /* # pixels in vertical-direction */
  int n;               /* # pixels in horizontal-direction */
}
image;


void import_JPEG_file (const char* filename, unsigned char** image_chars,
                       int* image_height, int* image_width,
                       int* num_components);
void export_JPEG_file (const char* filename, const unsigned char* image_chars,
                       int image_height, int image_width,
                       int num_components, int quality);

void allocate_image(image *u, int m, int n)
{
 (*u).image_data = (float**)malloc(m*sizeof(float*));

 for (int i = 0; i < m; i++) {
   (*u).image_data[i] = (float*)malloc(n*sizeof(float));

   for (int j = 0; j < n; j++)
    (*u).image_data[i][j] = 0;  // initialize
 }
} // end of allocate_image

void deallocate_image(image *u)
{

  for (int i = 0; i < (*u).m; i++) free((*u).image_data[i]);

  free((*u).image_data);

} // end of deallocate_image

void convert_jpeg_to_image(const unsigned char *image_chars, image *u)
{
  int i, j, m = (*u).m, n = (*u).n;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      (*u).image_data[i][j] = (float)image_chars[i*n + j];

}

void convert_image_to_jpeg(const image *u, unsigned char *image_chars)
{
  int i, j, m = (*u).m, n = (*u).n;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      image_chars[i*n + j] = (unsigned char)(*u).image_data[i][j];
}


int main(int argc, char *argv[])
{
  int i, j, k;
  int m, n, c, iters;
  int my_m, my_n, my_rank, num_procs;
  int my_mn[2];
  int my_m_start, my_m_stop, my_n_start, my_n_stop;
  float kappa;
  image u, u_bar, whole_image;
  unsigned char *image_chars, *my_image_chars;
  char *input_jpeg_filename, *output_jpeg_filename;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
  MPI_Request *request;

  input_jpeg_filename = argv[1];
  output_jpeg_filename = argv[2];

  // kappa = atof(argv[3]);
  // iters = atoi(argv[4]);

  kappa = 0.2;
  iters = 100;

  if (my_rank==0){
    import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
    allocate_image (&whole_image, m, n);
  }

  MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);


  if ((my_rank == num_procs - 1) && (m%num_procs != 0)){
    my_m = (m - 2)/num_procs + m%num_procs;
  }
  else{
    my_m = m/num_procs;
  }

  if ((my_rank == num_procs - 1) && (n%num_procs != 0)){
    my_n = (n - 2)/num_procs + n%num_procs;
  }
  else{
    my_n = n/num_procs;
  }

  my_image_chars = (unsigned char*)malloc(my_m*my_n*sizeof(unsigned char));

  allocate_image(&u, my_m, my_n);
  allocate_image(&u_bar, my_m, my_n);


  if (my_rank != 0){
    my_mn[0] = my_m;
    my_mn[1] = my_n;
    MPI_Send(my_mn, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Irecv(&my_image_chars, my_m*my_n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, request); // receive pixel data
  }

  else{
    int counter = 0;
    for (k = 1; k < num_procs - 1; k++){
      MPI_Recv(my_mn, 1, MPI_INT, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      my_image_chars = (unsigned char*)malloc(my_mn[0]*my_mn[1]*sizeof(unsigned char));

      my_m_start = my_rank*my_mn[0];
      my_m_stop = my_m_start + my_mn[0];

      my_n_start = my_rank*my_mn[1];
      my_n_stop = my_n_start + my_mn[1];

      for (i = my_m_start; i < my_m_stop; i++){
        for (j = my_n_start; j < my_n_stop; j++){
          my_image_chars[counter] = image_chars[i*n + j];
          counter++;
        }
      }

      MPI_Send(my_image_chars, my_mn[0]*my_mn[1], MPI_INT, i, 0, MPI_COMM_WORLD);
      free(my_image_chars);
    }
  }





}
