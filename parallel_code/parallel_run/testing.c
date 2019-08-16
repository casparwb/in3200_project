#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

int main(int nargs, char **args){
  int size, my_rank, i = 0;
  int m = 4289; int n = 2835;
  int my_m, my_n;
  int my_m_start, my_n_start, my_m_stop, my_n_stop;
  MPI_Init (&nargs, &args);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Status *status;

  MPI_Comm comm;
  int dim[2], period[2], reorder;
  int coord[2], id;

  dim[0] = sqrt(size); dim[1] = sqrt(size);
  period[0] = 0; period[1] = 0;

  reorder = 1;

  // MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &comm);
  //
  //
  // MPI_Cart_coords(comm, my_rank, 2, coord);
  // printf("Rank %d coordinates are %d %d\n", my_rank, coord[0], coord[1]);

  if ((my_rank == size - 1) && (m%size != 0)){
    my_m = (m - 2)/size + m%size;
  }
  else{
    my_m = m/size;
  }

  if ((my_rank == size - 1) && (n%size != 0)){
    my_n = (n - 2)/size + n%size;
  }
  else{
    my_n = n/size;
  }

  // printf("my_m = %d\n", my_m);
  // printf("my_n = %d\n", my_n);

  my_m_start = my_rank*my_m;
  my_m_stop = my_m_start + my_m;

  my_n_start = my_rank*my_n;
  my_n_stop = my_n_start + my_n;


  // printf("My_m_start = %d\n", my_m_start);
  // printf("My_m_stop = %d\n", my_m_stop);
  // printf("My_n_start = %d\n", my_n_start);
  // printf("My_n_stop = %d\n", my_n_stop);
  int test[2];
  test[0] = 0; test[1] = 1;


  if (my_rank == 0){
    test[0] = 3; test[1] = 6;
    MPI_Send(&test, 2, MPI_INT, 1, 0, MPI_COMM_WORLD);
  }
  else if (my_rank == 1){
    MPI_Recv(&test, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("%d %d\n", test[0], test[1]);
  }



  MPI_Finalize();
  return 0;
}
