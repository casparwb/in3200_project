#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[])
{

  int my_rank, num_procs, my_grid_rank;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);

  MPI_Request request[num_procs];
  MPI_Status status[num_procs];

  MPI_Comm grid_comm;
  int dim[2], period[2], reorder;
  int coord[2];
  int up, down, left, right;

  dim[0] = 2; dim[1] = 2;

  dim[0] = sqrt(num_procs); dim[1] = sqrt(num_procs);
  period[0] = 0; period[1] = 0;

  reorder = 1;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &grid_comm);
  MPI_Cart_coords(grid_comm, my_rank, 2, coord);
  MPI_Comm_rank (grid_comm, &my_grid_rank);

  printf("My rank is %d and I have coordinates (%d %d)\n", my_grid_rank, coord[0], coord[1]);

  if (my_grid_rank == 6){
    MPI_Cart_shift(grid_comm, 0, 1, &down, &up);
    printf("Up: %d |Down: %d \n", up, down);
    MPI_Cart_shift(grid_comm, 1, 1, &left, &right);
    printf("Left: %d |Right: %d \n", left, right);
    // if (src_rank != MPI_PROC_NULL)
    //   printf("My rank: %d | Source rank: %d, Dest rank: %d\n", my_grid_rank, src_rank, dest_rank);
    // if (dest_rank != MPI_PROC_NULL)
    //   printf("My rank: %d | Source rank: %d, Dest rank: %d\n", my_grid_rank, src_rank, dest_rank);

  }


  MPI_Finalize();
  return 0;
}

/*
MPI_Cart_shit(comm, 0, 1, &up, &down);
MPI_Cart_shift(comm, 1, 1, &left, &right);
*/


/*
My rank is 0 and I have coordinates (0 0)
My rank is 1 and I have coordinates (0 1)
My rank is 2 and I have coordinates (0 2)
My rank is 4 and I have coordinates (1 1)
Up: 1 |Down: 7
Left: 3 |Right: 5
My rank is 5 and I have coordinates (1 2)
My rank is 6 and I have coordinates (2 0)
My rank is 7 and I have coordinates (2 1)
My rank is 8 and I have coordinates (2 2)
My rank is 3 and I have coordinates (1 0)
*/
