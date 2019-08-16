#include <mpi.h>

typedef struct{
  float **image_data;  /* a 2D array of floats */
  int m;               /* # pixels in vertical-direction */
  int n;               /* # pixels in horizontal-direction */
}
image;

void allocate_image(image *u)
{
  int i, j, m = (*u).m, n = (*u).n;

 (*u).image_data = (float**)malloc(m*sizeof(float*));

 for (i = 0; i < m; i++) {
   (*u).image_data[i] = (float*)malloc(n*sizeof(float));

   for (j = 0; j < n; j++)
    (*u).image_data[i][j] = 0;  // initialize
 }
} // end of allocate_image

void deallocate_image(image *u)
{
  int m = (*u).m;
  int i;
  for (i = 0; i < m; i++){
    free((*u).image_data[i]);
  }
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
  /* function for packing column-stored data */
  int i, m = (*u).m, n = (*u).n;

  if (left_right == 1){ // left column
    for (i = 0; i < m; i++){
      buf[i] = (*u).image_data[i][0];
    }
  }

  else if (left_right == 0){ // right_column
    for (i = 0; i < m; i++){
      buf[i] = (*u).image_data[i][n - 1];
    }
  }

} // end of packsendbuffer


void iso_diffusion_denoising_parallel(image *u, image*u_bar, float kappa, int iters, MPI_Comm comm)
{

  int i, j, k, my_m = (*u).m, my_n = (*u).n;
  float **temp_ptr;
  int my_rank, coord[2];
  MPI_Request req_x1, req_x2, req_y1, req_y2;
  MPI_Status stat_x1, stat_x2, stat_y1, stat_y2;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Cart_coords(comm, my_rank, 2, coord);

  int right, left, up, down; // neighbours
  float *sleft, *sright; // buffers for halo values
  float *recvup, *recvdown, *recvleft, *recvright;

  MPI_Cart_shift(comm, 1, 1, &left, &right);
  MPI_Cart_shift(comm, 0, 1, &up, &down);

  if (left != MPI_PROC_NULL){
    sleft =     (float*)malloc(my_m*sizeof(float));
    recvleft =  (float*)malloc(my_m*sizeof(float));
  }

  if (right != MPI_PROC_NULL){
    sright =    (float*)malloc(my_m*sizeof(float));
    recvright = (float*)malloc(my_m*sizeof(float));
  }

  if (up != MPI_PROC_NULL){
    recvup =    (float*)malloc(my_n*sizeof(float));
  }

  if (down != MPI_PROC_NULL){
    recvdown =  (float*)malloc(my_n*sizeof(float));
  }

  for (k = 0; k < iters; k++){

    /*x-direction*/
    // MPI_Cart_shift(comm, 1, 1, &left, &right);
    if (left != MPI_PROC_NULL){
      // printf("My rank is %d, my coords are (%d %d) and my left neighbour is %d\n", my_rank, coord[0], coord[1], left);
      MPI_Irecv(recvleft, my_m, MPI_FLOAT, left, 1, comm, &req_x1);
    }

    if (right != MPI_PROC_NULL){
      // printf("My rank is %d, my coords are (%d %d) and my right neighbour is %d\n", my_rank, coord[0], coord[1], right);
      MPI_Irecv(recvright, my_m, MPI_FLOAT, right, 0, comm, &req_x2);
    }

    if (right != MPI_PROC_NULL){
      packsendbuffer(u, sright, 0);
      MPI_Send(sright, my_m, MPI_FLOAT, right, 1, comm);
    }

    if (left != MPI_PROC_NULL){
      packsendbuffer(u, sleft, 1);
      MPI_Send(sleft, my_m, MPI_FLOAT, left, 0, comm);
    }

    if (left != MPI_PROC_NULL){
      MPI_Wait(&req_x1, &stat_x1);
    }
    if (right != MPI_PROC_NULL){
      MPI_Wait(&req_x2, &stat_x2);
    }

    /*y-direction*/
    // MPI_Cart_shift(comm, 0, 1, &up, &down);
    if (up != MPI_PROC_NULL){
      // printf("My rank is %d, my coords are (%d %d) and my upper neighbour is %d\n", my_rank, coord[0], coord[1], up);
      MPI_Irecv(recvup, my_n, MPI_FLOAT, up, 2, comm, &req_y1);
    }

    if (down != MPI_PROC_NULL){
      // printf("My rank is %d, my coords are (%d %d) and my upper neighbour is %d\n", my_rank, coord[0], coord[1], up);
      MPI_Irecv(recvdown, my_n, MPI_FLOAT, down, 3, comm, &req_y2);
    }

    if (up != MPI_PROC_NULL){
      MPI_Send((*u).image_data[1], my_n, MPI_FLOAT, up, 3, comm);
    }

    if (down != MPI_PROC_NULL){
      MPI_Send((*u).image_data[my_m - 2], my_n, MPI_FLOAT, down, 2, comm);
    }

    if (up != MPI_PROC_NULL)
      MPI_Wait(&req_y1, &stat_y1);

    if (down != MPI_PROC_NULL)
      MPI_Wait(&req_y2, &stat_y2);

      if (my_rank == 0){
        for (i = 1; i < my_m - 1; i++)
          for (j = 1; j < my_n - 1; j++)
            (*u_bar).image_data[i][j] =   (*u).image_data[i][j] +
                                        kappa*
                                        ( (*u).image_data[i-1][j] +
                                          (*u).image_data[i][j-1]-
                                        4*(*u).image_data[i][j] +
                                          (*u).image_data[i][j+1] +
                                          (*u).image_data[i+1][j]);

        // halo point [i][my_n - 1]:
        for (i = 1; i < my_m - 1; i++){
          (*u_bar).image_data[i][my_n - 1] =  (*u).image_data[i][my_n - 1] +
                                              kappa*
                                            ( (*u).image_data[i-1][my_n - 1] +
                                              (*u).image_data[i][my_n - 2]-
                                            4*(*u).image_data[i][my_n - 1] +
                                              recvright[i] +
                                              (*u).image_data[i+1][my_n - 1]);
        }

        // halo point [my_m - 1][j]:
        for (j = 1; j < my_n - 1; j++){
          (*u_bar).image_data[my_m - 1][j] =  (*u).image_data[my_m - 1][j] +
                                              kappa*
                                            ( (*u).image_data[my_m - 2][j] +
                                              (*u).image_data[my_m - 1][j - 1]-
                                            4*(*u).image_data[my_m - 1][j] +
                                              (*u).image_data[my_m - 1][j+1] +
                                              recvdown[j]);
        }

      // single halo point [my_m - 1][my_n - 1]
      (*u_bar).image_data[my_m - 1][my_n - 1] = (*u).image_data[my_m - 1][my_n - 1] +
                                              kappa*
                                              ( (*u).image_data[my_m - 2][my_n - 1] +
                                                (*u).image_data[my_m - 1][my_n - 2]-
                                              4*(*u).image_data[my_m - 1][my_n - 1] +
                                              recvright[my_m - 1] +
                                              recvdown[my_n - 1]);

      } // end of if my_rank == 0

      else if (my_rank == 1){

        // inner points:
        for (i = 1; i < my_m - 1; i++)
          for (j = 1; j < my_n - 1; j++)
            (*u_bar).image_data[i][j] =   (*u).image_data[i][j] +
                                        kappa*
                                        ( (*u).image_data[i-1][j] +
                                          (*u).image_data[i][j-1]-
                                        4*(*u).image_data[i][j] +
                                          (*u).image_data[i][j+1] +
                                          (*u).image_data[i+1][j]);


        // halo point [i][0]
        for (i = 1; i < my_m - 1; i++){
          (*u_bar).image_data[i][0] =   (*u).image_data[i][0] +
                                      kappa*
                                      ( (*u).image_data[i-1][0] +
                                        recvleft[i]-
                                      4*(*u).image_data[i][0] +
                                        (*u).image_data[i][1] +
                                        (*u).image_data[i+1][0]);
        }

        // halo point [my_m - 1][j]
        for (j = 1; j < my_n - 1; j++){
          (*u_bar).image_data[my_m - 1][j] =   (*u).image_data[my_m - 1][j] +
                                      kappa*
                                      ( (*u).image_data[my_m - 2][j] +
                                        (*u).image_data[my_m - 1][j-1]-
                                      4*(*u).image_data[my_m - 1][j] +
                                        (*u).image_data[my_m - 1][j+1] +
                                        recvdown[j]);
        }

        // single point [my_m - 1][0]
        (*u_bar).image_data[my_m - 1][0] = (*u).image_data[my_m - 1][0] +
                                    kappa*
                                    ( (*u).image_data[i-1][0] +
                                      recvleft[my_m - 1]-
                                    4*(*u).image_data[my_m - 1][0] +
                                      (*u).image_data[my_m - 1][j+1] +
                                      recvdown[0]);
      } // end of if my_rank == 1


    else if (my_rank == 2){
      // inner points:

      for (i = 1; i < my_m - 1; i++)
        for (j = 1; j < my_n - 1; j++)
          (*u_bar).image_data[i][j] =   (*u).image_data[i][j] +
                                      kappa*
                                      ( (*u).image_data[i-1][j] +
                                        (*u).image_data[i][j-1]-
                                      4*(*u).image_data[i][j] +
                                        (*u).image_data[i][j+1] +
                                        (*u).image_data[i+1][j]);

      // halo point [0][j]
      for (j = 0; j < my_n - 1; j++){
        (*u_bar).image_data[0][j] =   (*u).image_data[0][j] +
                                    kappa*
                                    ( recvup[j] +
                                      (*u).image_data[0][j-1]-
                                    4*(*u).image_data[0][j] +
                                      (*u).image_data[0][j+1] +
                                      (*u).image_data[1][j]);
      }

      // halo point [i][my_n - 1]
      for (i = 1; i < my_m - 1; i++){
        (*u_bar).image_data[i][my_n - 1] =   (*u).image_data[i][my_n - 1] +
                                            kappa*
                                            ( (*u).image_data[i-1][my_n - 1] +
                                              (*u).image_data[i][my_n - 2]-
                                            4*(*u).image_data[i][my_n - 1] +
                                              recvright[i] +
                                              (*u).image_data[i+1][my_n - 1]);

      }
      // single halo point [0][my_n - 1]
      (*u_bar).image_data[0][my_n - 1] =   (*u).image_data[0][my_n - 1] +
                                  kappa*
                                  ( recvup[my_n - 1] +
                                    (*u).image_data[0][my_n - 2]-
                                  4*(*u).image_data[0][my_n - 1] +
                                    recvright[0] +
                                    (*u).image_data[1][my_n - 1]);
    } // end of if my_rank == 2

    else if (my_rank == 3){
      // inner points
      for (i = 1; i < my_m - 1; i++)
        for (j = 1; j < my_n - 1; j++)
          (*u_bar).image_data[i][j] =   (*u).image_data[i][j] +
                                      kappa*
                                      ( (*u).image_data[i-1][j] +
                                        (*u).image_data[i][j-1]-
                                      4*(*u).image_data[i][j] +
                                        (*u).image_data[i][j+1] +
                                        (*u).image_data[i+1][j]);


      // halo points [0][j]
      for (j = 1; j < my_n - 1; j++){
        (*u_bar).image_data[0][j] =   (*u).image_data[0][j] +
                                    kappa*
                                    ( recvup[j] +
                                      (*u).image_data[0][j-1]-
                                    4*(*u).image_data[0][j] +
                                      (*u).image_data[0][j+1] +
                                      (*u).image_data[1][j]);
      }

      // halo points [i][0]
      for (i = 1; i < my_m - 1; i++){
        (*u_bar).image_data[i][0] =   (*u).image_data[i][0] +
                                    kappa*
                                    ( (*u).image_data[i-1][0] +
                                      recvleft[i]-
                                    4*(*u).image_data[i][0] +
                                      (*u).image_data[i][1] +
                                      (*u).image_data[i+1][0]);
      }

      // single halo point [0][0]
      (*u_bar).image_data[0][0] =   (*u).image_data[0][0] +
                                  kappa*
                                  ( recvup[my_n - 1] +
                                    recvleft[my_m - 1]-
                                  4*(*u).image_data[0][0] +
                                    (*u).image_data[0][1] +
                                    (*u).image_data[1][0]);
    } // end of if my_rank == 3

    // pointer swapping
    temp_ptr = (*u_bar).image_data;
    (*u_bar).image_data = (*u).image_data;
    (*u).image_data = temp_ptr;
  } // end of iters-loop


  // if (left != MPI_PROC_NULL)
  //   free(sleft); free(recvleft);
  //
  // if (right != MPI_PROC_NULL)
  //   free(sright); free(recvright);
  //
  // if (up != MPI_PROC_NULL)
  //   free(recvup);
  //
  // if (down != MPI_PROC_NULL)
  //   free(recvdown);

} // end of iso-function
