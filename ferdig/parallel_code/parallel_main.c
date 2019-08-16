#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "functions.h"



void import_JPEG_file (const char* filename, unsigned char** image_chars,
                       int* image_height, int* image_width,
                       int* num_components);
void export_JPEG_file (const char* filename, const unsigned char* image_chars,
                       int image_height, int image_width,
                       int num_components, int quality);


int main(int argc, char *argv[])
{
  int i, j, k;
  int m, n, c, iters;
  int my_m, my_n, my_rank, num_procs, my_grid_rank;
  int my_mn[2];
  int my_m_start, my_m_stop, my_n_start, my_n_stop;
  int counter;
  float kappa;
  double t;
  image u, u_bar, whole_image;
  unsigned char *image_chars, *my_image_chars;
  char *input_jpeg_filename, *output_jpeg_filename;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);

  if (num_procs != 4){
    printf("Please run with 4 processors\n");
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Request request;
  MPI_Status status;

  MPI_Comm grid_comm;
  int dim[2], period[2], reorder;
  int coord[2];

  dim[0] = 2; dim[1] = 2;
  period[0] = 0; period[1] = 0;

  reorder = 1;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &grid_comm);
  MPI_Comm_rank (grid_comm, &my_grid_rank);
  MPI_Cart_coords(grid_comm, my_grid_rank, 2, coord);


  input_jpeg_filename = argv[1];
  output_jpeg_filename = argv[2];

  kappa = atof(argv[3]);
  iters = atoi(argv[4]);


  if (my_grid_rank==0){
    import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
    whole_image.m = m; whole_image.n = n;
    allocate_image (&whole_image);
    convert_jpeg_to_image(image_chars, &whole_image);
    printf("Import successful! Image dimensions: %d x %d \n", m, n);
  }

  MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);


  /*Dividing the data*/
  if (my_grid_rank == 0){
    my_m = m/dim[0] + m%2;
    my_n = n/dim[1] + n%2;
  }
  if (my_grid_rank == 1){
    my_m = m/dim[0] + m%2;
    my_n = n/dim[1];
  }
  if (my_grid_rank == 2){
    my_m = m/dim[0];
    my_n = n/dim[1] + n%2;
  }
  else if (my_grid_rank == 3){
    my_m = m/dim[0];
    my_n = n/dim[1];
  }

  u.m = my_m;
  u.n = my_n;

  u_bar.m = my_m;
  u_bar.n = my_n;

  allocate_image(&u);
  allocate_image(&u_bar);

  /*Partioning image_chars to each processor*/
  if (my_grid_rank != 0){
    my_mn[0] = my_m;
    my_mn[1] = my_n;
    MPI_Send(my_mn, 2, MPI_INT, 0, 0, grid_comm); // sending my_m and my_n

    my_image_chars = (unsigned char*)malloc(my_m*my_n*sizeof(unsigned char));

    MPI_Recv(my_image_chars, my_m*my_n, MPI_UNSIGNED_CHAR, 0, 0, grid_comm, MPI_STATUS_IGNORE);
  }
  else if (my_grid_rank == 0)
  {
    int proc_coord[2]; // for storing coordinates of each proc rank
    unsigned char *buffer;

    /* processor 0 */
    my_image_chars = (unsigned char*)malloc(my_m*my_n*sizeof(unsigned char));
    my_m_start = coord[0]*my_m;
    my_m_stop = (coord[0] + 1)*my_m;

    my_n_start = coord[1]*my_n;
    my_n_stop = (coord[1] + 1)*my_n;

    counter = 0;
    for (i = my_m_start; i < my_m_stop; i++){
      for (j = my_n_start; j < my_n_stop; j++){
        my_image_chars[counter] = image_chars[i*n + j];
        counter++;
      }
    }


    // everybody else
    for (k = 1; k < num_procs; k++)
    {
      MPI_Recv(my_mn, 2, MPI_INT, k, 0, grid_comm, MPI_STATUS_IGNORE);          // receive my_m and my_n for each processor
      MPI_Cart_coords(grid_comm, k, 2, proc_coord);

      buffer = (unsigned char*)malloc(my_mn[0]*my_mn[1]*sizeof(unsigned char)); // allocate array for packing and sending image_chars

      my_m_start = proc_coord[0]*my_mn[0];
      my_m_stop = (proc_coord[0] + 1)*my_mn[0];

      my_n_start = proc_coord[1]*my_mn[1];
      my_n_stop = (proc_coord[1] + 1)*my_mn[1];


      counter = 0;
      for (i = my_m_start; i < my_m_stop; i++){
        for (j = my_n_start; j < my_n_stop; j++){
          buffer[counter] = image_chars[i*n + j];
          counter++;
        }
      }

      MPI_Send(buffer, my_mn[0]*my_mn[1], MPI_UNSIGNED_CHAR, k, 0, grid_comm);
      free(buffer);

    } // end of for-loop

  } // end of if

  convert_jpeg_to_image(my_image_chars, &u);
  convert_jpeg_to_image(my_image_chars, &u_bar);

  t = MPI_Wtime();
  iso_diffusion_denoising_parallel (&u, &u_bar, kappa, iters, grid_comm);
  t = MPI_Wtime() - t;


  /*Collecting the denoised image data*/
  if (my_grid_rank != 0){

    float *buffer = (float*)malloc(my_m*my_n*sizeof(float)); // buffer for packing and sending data

    /* Packing data */
    counter = 0;
    for (i = 1; i < my_m; i++)
      for (j = 1; j < my_n; j++){
        buffer[counter] = u.image_data[i][j];
        counter++;
      }
    MPI_Send(my_mn, 2, MPI_INT, 0, 0, grid_comm);
    MPI_Send(buffer, my_m*my_n, MPI_FLOAT, 0, 0, grid_comm);

    deallocate_image(&u);
    deallocate_image(&u_bar);
    free(buffer);

  }
  else if (my_grid_rank == 0){
    int proc_coord[2];
    float *buffer = (float*)malloc(m*n*sizeof(float));

    /* Collect data from proc 0 */

    my_m_start = coord[0]*my_m;
    my_m_stop = (coord[0] + 1)*my_m;

    my_n_start = coord[1]*my_n;
    my_n_stop = (coord[1] + 1)*my_n;


    for (i = my_m_start; i < my_m_stop; i++)
      for (j = my_n_start; j < my_n_stop; j++)
        whole_image.image_data[i][j] = u.image_data[i][j];

    // collect data from everyone else
    for (k = 1; k < num_procs; k++){
      MPI_Recv(my_mn, 2, MPI_INT, k, 0, grid_comm, MPI_STATUS_IGNORE);
      MPI_Recv(buffer, my_mn[0]*my_mn[1], MPI_FLOAT, k, 0, grid_comm, MPI_STATUS_IGNORE);
      MPI_Cart_coords(grid_comm, k, 2, proc_coord);

      my_m_start = proc_coord[0]*my_mn[0];
      my_m_stop = (proc_coord[0] + 1)*my_mn[0];

      my_n_start = proc_coord[1]*my_mn[1];
      my_n_stop = (proc_coord[1] + 1)*my_mn[1];


      counter = 0;
      for (i = my_m_start + 1; i < my_m_stop; i++){
        for (j = my_n_start + 1; j < my_n_stop; j++){
          whole_image.image_data[i][j] = buffer[counter];
          counter++;
        }
      }

    } // end of outer for
    free(buffer);


    convert_image_to_jpeg(&whole_image, image_chars);
    export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
    deallocate_image(&whole_image);
    deallocate_image(&u);
    deallocate_image(&u_bar);
  } // end of else

  printf("My rank: %d | Iso parallel time: %lfms\n", my_grid_rank, t*1000);

  double total_time;
  MPI_Allreduce(&t, &total_time, 1, MPI_DOUBLE, MPI_MAX, grid_comm);
  if (my_grid_rank == 0){
    printf("--------------------------------------\n");
    printf("Global time taken by iso_denoising: %lfs\n", total_time*1000);
  }

  free(my_image_chars);
  MPI_Finalize();
  return 0;

} // end of main function
