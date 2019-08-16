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

void packsendbuffer(image *u, float *buf, int left_right){

  int i, m = (*u).m, n = (*u).n;
  if (left_right == 0){ // left column
    for (i = 0; i < m; i++){
      buf[i] = (*u).image_data[i][0];
    }
  }

  else if (left_right == 1){ // right_column
    for (i = 0; i < m; i++){
      buf[i] = (*u).image_data[i][n - 1];
    }
  }


} // end of packsendbuffer

void checkborder(int num_procs, int coords, int *bottomleft, int *bottomright, int *topleft, int *topright, int *border, int *corner, int dim){
  /*Check border*/
  if ((coords[0] == 0 || coords[0] == dim[0] - 1) && (coords[1] == 0 || coords[1] == dim[1] - 1))
    *corner = 1;
  if ((coords[0] == 0 || coords[0] == dim[0] - 1) || (coords[1] == 0 || coords[1] == dim[1] - 1))
    *border = 1;

  for (int k = 1; k < num_procs; k++){
    if (border = TRUE && coord[0] == 0 && coord[1] == 0){
      *bottomleft = 1;
    }
    if (border = TRUE && coord[0] == 0 && coord[1] == (dim[1] - 1)){
      *bottomright = 1;
    }
    if (border = TRUE && coord[0] == (dim[0] - 1) && coord[1] == 0){
      *topleft = 1;
    }
    if (border = TRUE && coord[0] == (dim[0] - 1) && coord[1] = (dim[1] - 1)){
      *topright = 1;
    }
  }

}

void iso_diffusion_denoising_parallel(image *u, image*u_bar, float kappa, int iters, MPI_Comm comm)
{


  int i, j, k, my_m = (*u).m, my_n = (*u).n;
  float **temp_ptr;
  int my_rank;
  MPI_Request req_x1, req_x2, req_y1, req_y2;
  MPI_Status stat_x1, stat_x2, stat_y1, stat_y2;
  MPI_Comm_rank(comm, &my_rank);

  int right, left, up, down; // neighbours
  float *sleft, *sright; // buffers for halo values
  float *recvup, *recvdown, *recvleft, *recvright;

  // sendbuffer up, down is not required
  sleft = (float*)malloc((my_m + 1)*sizeof(float));
  sright = (float*)malloc((my_m + 1)*sizeof(float));

  recvup = (float*)malloc((my_n + 1)*sizeof(float));
  recvdown = (float*)malloc((my_n + 1)*sizeof(float));
  recvleft = (float*)malloc((my_m + 1)*sizeof(float));
  recvright = (float*)malloc((my_m + 1)*sizeof(float));

  for (k = 0; k < iters; k++){

    /*x-direction*/
    MPI_Cart_shift(comm, 1, 1, &left, &right);
    if (left != MPI_PROC_NULL){
      // printf("My rank is %d, my coords are (%d %d) and my left neighbour is %d\n", my_grid_rank, coord[0], coord[1], left);
      MPI_Irecv(recvleft, my_m + 1, MPI_FLOAT, left, 0, comm, &req_x1);
    }

    if (right != MPI_PROC_NULL){
      // printf("My rank is %d, my coords are (%d %d) and my right neighbour is %d\n", my_grid_rank, coord[0], coord[1], right);
      MPI_Irecv(recvright, my_m + 1, MPI_FLOAT, right, 0, comm, &req_x2);
    }

    if (left != MPI_PROC_NULL){
      packsendbuffer(u, sleft, 0);
      MPI_Send(sleft, my_m, MPI_FLOAT, left, 0, comm);

    }
    if (right != MPI_PROC_NULL){
      packsendbuffer(u, sright, 1);
      MPI_Send(sright, my_m, MPI_FLOAT, right, 0, comm);
    }

    if (left != MPI_PROC_NULL){
      MPI_Wait(&req_x1, &stat_x1);
    }
    if (right != MPI_PROC_NULL){
      MPI_Wait(&req_x2, &stat_x2);
    }

    /*y-direction*/
    MPI_Cart_shift(comm, 0, 1, &up, &down);
    if (up != MPI_PROC_NULL){
      // printf("My rank is %d, my coords are (%d %d) and my upper neighbour is %d\n", my_grid_rank, coord[0], coord[1], up);
      MPI_Irecv(recvup, my_n + 1, MPI_FLOAT, up, 0, comm, &req_y1);
    }

    if (down != MPI_PROC_NULL){
      // printf("My rank is %d, my coords are (%d %d) and my upper neighbour is %d\n", my_grid_rank, coord[0], coord[1], up);
      MPI_Irecv(recvdown, my_n + 1, MPI_FLOAT, down, 0, comm, &req_y2);
    }

    if (up != MPI_PROC_NULL){
      MPI_Send((*u).image_data[0], my_n, MPI_FLOAT, up, 0, comm);
    }

    if (down != MPI_PROC_NULL){
      MPI_Send((*u).image_data[my_m - 1], my_n, MPI_FLOAT, down, 0, comm);
    }

    if (up != MPI_PROC_NULL)
      MPI_Wait(&req_y1, &stat_y1);

    if (down != MPI_PROC_NULL)
      MPI_Wait(&req_y2, &stat_y2);


    for (i = 1; i < my_m - 1; i++)
      for (j = 1; j < my_n - 1; j++)
        (*u_bar).image_data[i][j] =   (*u).image_data[i][j] +
                                    kappa*
                                    ( (*u).image_data[i-1][j] +
                                      (*u).image_data[i][j-1]-
                                    4*(*u).image_data[i][j] +
                                      (*u).image_data[i][j+1] +
                                      (*u).image_data[i+1][j]);


    /*calculating border points minus corner:*/
    for (i = 1; i < my_m - 1; i++){
      (*u_bar).image_data[i][0] = (*u).image_data[i][0] +
                                  kappa*
                                  ( (*u).image_data[i-1][0] +
                                    recvleft[i] - // halo point
                                  4*(*u).image_data[i][0] +
                                    (*u).image_data[i][1] +
                                    (*u).image_data[i+1][0]);



      (*u_bar).image_data[i][my_n - 1] = (*u).image_data[i][my_n - 1] +
                                  kappa*
                                  ( (*u).image_data[i-1][my_n - 1] +
                                    (*u).image_data[i][my_n - 2] -
                                  4*(*u).image_data[i][my_n - 1] +
                                    recvright[i] +
                                    (*u).image_data[i+1][my_n - 1]);

    }
    for (j = 1; j < my_n - 1; j++){
      (*u_bar).image_data[0][j] = (*u).image_data[0][j] +
                                  kappa*
                                  ( recvup[i] +
                                    (*u).image_data[0][j-1] -
                                  4*(*u).image_data[0][j] +
                                    (*u).image_data[0][j+1] +
                                    (*u).image_data[1][my_n - 1]);


      (*u_bar).image_data[my_m - 1][j] = (*u).image_data[my_m - 1][j] +
                                         kappa*
                                       ( (*u).image_data[my_m - 2][j] +
                                         (*u).image_data[my_m - 1][j-1]-
                                       4*(*u).image_data[my_m - 1][j] +
                                         (*u).image_data[my_m - 1][j+1] +
                                         recvdown[i]);
    }

    /*calculating corners*/
    (*u_bar).image_data[0][0] =   (*u).image_data[0][0] +
                                  kappa*
                                ( recvdown[0] +
                                  recvleft[my_n - 1]-
                                4*(*u).image_data[0][0] +
                                  (*u).image_data[0][1] +
                                  (*u).image_data[1][0]);

    (*u_bar).image_data[0][my_n - 1] = (*u).image_data[0][my_n - 1] +
                                       kappa*
                                     ( recvdown[my_n - 1] +
                                       recvright[0]-
                                     4*(*u).image_data[0][0] +
                                       (*u).image_data[0][1] +
                                       (*u).image_data[1][0]);


    (*u_bar).image_data[my_m - 1][0] =  (*u).image_data[my_m - 1][0] +
                                      kappa*
                                      ( (*u).image_data[my_m - 1][j] +
                                      recvleft[my_m - 1]-
                                      4*(*u).image_data[my_m - 1][0] +
                                        (*u).image_data[my_m - 1][1] +
                                      recvup[0]);

    (*u_bar).image_data[my_m - 1][my_n - 1] =    (*u).image_data[my_m - 1][my_n - 1] +
                                               kappa*
                                               ( (*u).image_data[my_m - 2][my_n - 1] +
                                                 (*u).image_data[my_m - 1][my_n - 2] -
                                               4*(*u).image_data[my_m - 1][my_n - 1] +
                                               recvright[my_m - 1] +
                                               recvup[my_n - 1]);


    temp_ptr = (*u_bar).image_data;
    (*u_bar).image_data = (*u).image_data;
    (*u).image_data = temp_ptr;
  } // end of iters-loop



} // end of iso-function

int main(int argc, char *argv[])
{
  int i, j, k;
  int m, n, c, iters;
  int my_m, my_n, my_rank, num_procs, my_grid_rank;
  int my_mn[2];
  int my_m_start, my_m_stop, my_n_start, my_n_stop;
  int counter;
  int border, corner, bottomleft, bottomright, topleft, topright; // corners
  float kappa;
  image u, u_bar, whole_image;
  unsigned char *image_chars, *my_image_chars;
  char *input_jpeg_filename, *output_jpeg_filename;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);

  MPI_Request request;
  MPI_Status status;

  MPI_Comm grid_comm;
  int dim[2], period[2], reorder;
  int coord[2];

  dim[0] = sqrt(num_procs); dim[1] = sqrt(num_procs);
  period[0] = 0; period[1] = 0;

  reorder = 1;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &grid_comm);
  MPI_Cart_coords(grid_comm, my_rank, 2, coord);
  // printf("My rank is %d and I have coordinates (%d %d)\n", my_rank, coord[0], coord[1]);
  MPI_Comm_rank (grid_comm, &my_grid_rank);


  input_jpeg_filename = argv[1];
  output_jpeg_filename = argv[2];

  // kappa = atof(argv[3]);
  iters = atoi(argv[3]);

  kappa = 0.2;

  if (my_grid_rank==0){
    import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
    allocate_image (&whole_image, m, n);
    whole_image.m = m; whole_image.n = n;
    printf("%d | %d \n", m, n);
  }

  MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);



  if (my_rank%2){
    my_m = (m - 2)/dim[0] + (m - 2)%2;
    my_n = (n - 2)/dim[1] + (n - 2)%2;
  }
  else {
    my_m = (m - 2)/dim[0];
    my_n = (n - 2)/dim[1];
  }

  printf("My rank is %d and my_m: %d| my_n: %d\n", my_rank, my_m, my_n);

  u.m = my_m;
  u.n = my_n;

  checkborder(num_procs, coords, &bottomleft, &bottomright, &topleft, &topright, &border, &corner, dim);

  if (corner == 1){
    allocate_image(&u, my_m + 1, my_n + 1);
    allocate_image(&u_bar, my_m + 1, my_n + 1);
  }
  if (border == 1 && corner == 0){
    allocate_image(&u, my_m, my_n);
    allocate_image(&u_bar, my_m, my_n);

  }


  if (my_grid_rank != 0){
    my_mn[0] = my_m;
    my_mn[1] = my_n;
    MPI_Send(my_mn, 2, MPI_INT, 0, 0, grid_comm);
    MPI_Send(coord, 2, MPI_INT, 0, 0, grid_comm);

    my_image_chars = (unsigned char*)malloc(my_m*my_n*sizeof(unsigned char));

    MPI_Recv(my_image_chars, my_m*my_n, MPI_FLOAT, 0, 0, grid_comm, MPI_STATUS_IGNORE);
  }
  else if (my_grid_rank == 0)
  {
    int proc_coord[2];
    int counter;
    for (k = 1; k < num_procs; k++)
    {
      MPI_Recv(my_mn, 2, MPI_INT, k, 0, grid_comm, MPI_STATUS_IGNORE); // receive my_m and my_n for each processor
      MPI_Recv(proc_coord, 2, MPI_INT, k, 0, grid_comm, MPI_STATUS_IGNORE); // receive the coordinates of each processor
      my_image_chars = (unsigned char*)malloc(my_mn[0]*my_mn[1]*sizeof(unsigned char)); // allocate array for sending image_chars


      my_m_start = proc_coord[0]*my_m + 1;
      my_m_stop = (proc_coord[0] + 1)*my_m;

      my_n_start = proc_coord[1]*my_n + 1;
      my_n_stop = (proc_coord[1] + 1)*my_n;


      printf("My_rank: %d, my_m_start: %d, my_m_stop: %d my_n_start: %d, my_n_stop: %d\n",
            k, my_m_start, my_m_stop, my_n_start, my_n_stop);

      counter = 0;
      for (i = my_m_start; i <= my_m_stop; i++){
        for (j = my_n_start; j <= my_n_stop; j++){
          my_image_chars[counter] = image_chars[i*n + j];
          counter++;
        }
      }
      MPI_Send(my_image_chars, my_mn[0]*my_mn[1], MPI_UNSIGNED_CHAR, k, 0, grid_comm);
      free(my_image_chars);

    } // end of for-loop

    /*Allocating to processor 0:*/

    my_image_chars = (unsigned char*)malloc(my_m*my_n*sizeof(unsigned char)); // allocate array for sending image_chars
    my_m_start = coord[0]*my_m + 1;
    my_m_stop = (coord[0] + 1)*my_m;

    my_n_start = coord[1]*my_n + 1;
    my_n_stop = (coord[1] + 1)*my_n;

    counter = 0;
    for (i = my_m_start; i <= my_m_stop; i++){
      for (j = my_n_start; j <= my_n_stop; j++){
        my_image_chars[counter] = image_chars[i*n + j];
        counter++;
      }
    }
    printf("My_rank: %d, my_m_start: %d, my_m_stop: %d my_n_start: %d, my_n_stop: %d\n",
          my_rank, my_m_start, my_m_stop, my_n_start, my_n_stop);

  } // end of if

  convert_jpeg_to_image(my_image_chars, &u);
  printf("Rank %d successfully converted to image!\n", my_rank);

  // iso_diffusion_denoising_parallel (&u, &u_bar, kappa, iters, grid_comm);
  // printf("Rank %d successfully denoised image!\n", my_grid_rank);

  if (my_grid_rank != 0){

    float *buffer = (float*)malloc(my_m*my_n*sizeof(float));
    for (i = 0; i < my_m; i++)
      for (j = 0; j < my_n; j++){
        buffer[counter] = u_bar.image_data[i][j];
        counter++;
      }
    MPI_Isend(my_mn, 2, MPI_INT, 0, 0, grid_comm, &request);
    MPI_Isend(buffer, my_m*my_n, MPI_FLOAT, 0, 0, grid_comm, &request);
    // free(buffer);

  }
  else if (my_grid_rank == 0){
    int proc_coord[2];
    // float *buffer = (float*)malloc(((m - 2)/num_procs + 1)*((n - 2)/num_procs + 1)*sizeof(float));

    float *buffer = (float*)malloc(m*n*sizeof(float));

    for (k = 1; k < num_procs; k++){
      printf("iter nr %d\n", k);
      MPI_Recv(my_mn, 2, MPI_INT, k, 0, grid_comm, MPI_STATUS_IGNORE);
      printf("First receive succesful\n");
      MPI_Recv(buffer, my_mn[0]*my_mn[1], MPI_FLOAT, k, 0, grid_comm, MPI_STATUS_IGNORE);
      printf("Second receive succesful\n");
      MPI_Cart_coords(grid_comm, k, 2, proc_coord);

      my_m_start = proc_coord[0]*my_m + 1;
      my_m_stop = (proc_coord[0] + 1)*my_m;

      my_n_start = proc_coord[1]*my_n + 1;
      my_n_stop = (proc_coord[1] + 1)*my_n;

      counter = 0;
      for (i = my_m_start; i <= my_m_stop; i++){
        for (j = my_n_start; j <= my_n_stop; j++){
          whole_image.image_data[i][j] = buffer[counter];

          counter++;
        }
      }
    } // end of outer for

    my_m_start = coord[0]*my_m + 1;
    my_m_stop = (coord[0] + 1)*my_m;

    my_n_start = coord[1]*my_n + 1;
    my_n_stop = (coord[1] + 1)*my_n;


    counter = 0;
    for (i = my_m_start; i < my_m_stop; i++){
      for (j = my_n_start; j < my_n_stop; j++){
        whole_image.image_data[i][j] = u_bar.image_data[i][j];
        counter++;
      }
    }

    free(buffer);

    convert_image_to_jpeg(&whole_image, image_chars);
    export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
    // deallocate_image(&whole_image);
  } // end of else

  // deallocate_image(&u);
  // deallocate_image(&u_bar);
  MPI_Barrier(grid_comm);

  MPI_Finalize();
  return 0;

} // end of main function
