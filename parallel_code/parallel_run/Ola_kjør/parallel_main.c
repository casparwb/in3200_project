#include <stdio.h>
#include <stdlib.h>
#include "functions.c"
#include <math.h>
#include <mpi.h>

void import_JPEG_file(const char* filename, unsigned char** image_chars,
                      int* image_height, int* image_width,
                      int* num_components);

void export_JPEG_file(const char* filename, const unsigned char* image_chars,
                      int image_height, int image_width,
                      int num_components, int quality);

int main(int argc, char *argv[]) {
    int m, n, c, iters;
    int my_m, my_n, my_rank, num_procs, my_cart_rank;
    int my_m_start, my_m_stop, my_n_start, my_n_stop;
    int my_mn[2];
    int border, corner, bottom_left, bottom_right, top_left, top_right;

    float kappa;
    image u, u_bar, whole_image;
    unsigned char *image_chars, *my_image_chars;
    char *input_jpeg_filename, *output_jpeg_filename;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    kappa = atoi(argv[1]);
    iters = atoi(argv[2]);
    input_jpeg_filename = argv[3];
    output_jpeg_filename = argv[4];

    MPI_Request request;
    MPI_Status status;

    // setting up cartesian coordinates for processors
    MPI_Comm cart_comm;
    int dim[2]; int period[2]; int coord[2];
    dim[0] = 2; dim[1] = 2; //using 4 processors
    period[0] = 0; period[1] = 0;
    int reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, my_rank, 2, coord);
    printf("Rank %d has coordinates (%d %d)\n", my_rank, coord[0], coord[1]);
    MPI_Comm_rank(cart_comm, &my_cart_rank);

    if (my_cart_rank == 0) {
        import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
        allocate_image(&whole_image, m, n);
        whole_image.m = m; whole_image.n = n;
        printf("pixels: %d x %d \n", m, n);
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_cart_rank == 0) {
        my_m = m/dim[0] + m%2;
        my_n = n/dim[1] + n%2;
    }
    if (my_cart_rank == 1) {
        my_m = m/dim[0] + m%2;
        my_n = n/dim[1];
    }
    if (my_cart_rank == 2) {
        my_m = m/dim[0];
        my_n = n/dim[1] + n%2;
    }
    else if (my_cart_rank == 3){
      my_m = m/dim[0];
      my_n = n/dim[1];
    }

    allocate_image(&u, my_m, my_n);
    allocate_image(&u_bar, my_m, my_n);

    printf("Rank %d has my_m: %d| my_n: %d\n", my_cart_rank, my_m, my_n);

    u.m = my_m;
    u.n = my_n;

    int counter;
    if (my_cart_rank != 0) {
        my_mn[0] = my_m;
        my_mn[1] = my_n;
        MPI_Send(my_mn, 2, MPI_INT, 0, 0, cart_comm);
        MPI_Send(coord, 2, MPI_INT, 0, 1, cart_comm);

        my_image_chars = (unsigned char*)malloc((my_m*my_n)*sizeof(unsigned char));

        MPI_Recv(my_image_chars, my_m*my_n, MPI_UNSIGNED_CHAR, 0, 0, cart_comm,
                 MPI_STATUS_IGNORE);
    }
    else if (my_cart_rank == 0) {
        int proc_coord[2];
        unsigned char *buffer;

        for (int k = 1; k < num_procs; k++) {
            MPI_Recv(my_mn, 2, MPI_INT, k, 0, cart_comm, MPI_STATUS_IGNORE);
            MPI_Recv(proc_coord, 2, MPI_INT, k, 1, cart_comm,
                     MPI_STATUS_IGNORE);
            buffer = (unsigned char*)malloc((my_mn[0]*my_mn[1])*sizeof(unsigned
                      char));

            my_m_start = proc_coord[0]*my_mn[0];
            my_m_stop = (proc_coord[0] + 1)*my_mn[0];

            my_n_start = proc_coord[1]*my_mn[1];
            my_n_stop = (proc_coord[1] + 1)*my_mn[1];

            printf("My_rank: %d, my_m_start: %d, my_m_stop: %d my_n_start: %d, my_n_stop: %d\n",
                  k, my_m_start, my_m_stop, my_n_start, my_n_stop);

            counter = 0;
            for (int i = my_m_start; i < my_m_stop; i++) {
                for (int j = my_n_start; j < my_n_stop; j++) {
                    buffer[counter] = image_chars[i*n + j];
                    counter++;
                }
            }

            MPI_Send(buffer, my_mn[0]*my_mn[1], MPI_UNSIGNED_CHAR, k, 0,
                     cart_comm);
            free(buffer);
        }

        // setting up data for processor 0
        my_image_chars = (unsigned char*)malloc((my_m*my_n)*sizeof(unsigned
                          char));
        my_m_start = coord[0]*my_m;
        my_m_stop = (coord[0] + 1)*my_m;

        my_n_start = coord[1]*my_n;
        my_n_stop = (coord[1] + 1)*my_n;

        counter = 0;
        for (int i = my_m_start; i < my_m_stop; i++) {
            for (int j = my_n_start; j < my_n_stop; j++) {
                my_image_chars[counter] = image_chars[i*n + j];
                counter ++;
            }
        }

        printf("My_rank: %d, my_m_start: %d, my_m_stop: %d my_n_start: %d, my_n_stop: %d\n",
              my_rank, my_m_start, my_m_stop, my_n_start, my_n_stop);
    }

    convert_jpeg_to_image(my_image_chars, &u);
    printf("Rank %d successfully converted to image!\n", my_rank);

    iso_diffusion_denoising_parallel(&u, &u_bar, kappa, iters, cart_comm);
    printf("Rank %d successfully denoised image!\n", my_cart_rank);

    if (my_cart_rank != 0) {
        float *buffer = (float*)malloc((my_m*my_n)*sizeof(float));
        for (int i = 0; i < my_m; i++) {
            for (int j = 0; j < my_n; j++) {
                buffer[counter] = u_bar.image_data[i][j];
                counter++;
            }
        }

        MPI_Isend(my_mn, 2, MPI_INT, 0, 0, cart_comm, &request);
        MPI_Isend(buffer, my_m*my_n, MPI_FLOAT, 0, 0, cart_comm, &request);
        // free(buffer);
    }
    else if (my_cart_rank == 0) {
        int proc_coord[2];
        float *buffer = (float*)malloc((my_m*my_n)*sizeof(float));

        for (int k = 1; k < num_procs; k++) {
            printf("iter nr %d\n", k);
            MPI_Recv(my_mn, 2, MPI_INT, k, 0, cart_comm, MPI_STATUS_IGNORE);
            printf("First receive succesful\n");
            MPI_Recv(buffer, my_mn[0]*my_mn[1], MPI_FLOAT, k, 0, cart_comm,
                     MPI_STATUS_IGNORE);
            printf("Second receive succesful\n");
            MPI_Cart_coords(cart_comm, k, 2, proc_coord);

            my_m_start = proc_coord[0]*my_mn[0];
            my_m_stop = (proc_coord[0] + 1)*my_mn[0];

            my_n_start = proc_coord[1]*my_mn[1];
            my_n_stop = (proc_coord[1] + 1)*my_mn[1];

            counter = 0;
            for (int i = my_m_start; i < my_m_stop; i++){
              for (int j = my_n_start; j < my_n_stop; j++){
                whole_image.image_data[i][j] = buffer[counter];
                counter++;
              }
            }
        }
        free(buffer);

        my_m_start = coord[0]*my_m;
        my_m_stop = (coord[0] + 1)*my_m;

        my_n_start = coord[1]*my_n;
        my_n_stop = (coord[1] + 1)*my_n;

        for (int i = my_m_start; i < my_m_stop; i++)
          for (int j = my_n_start; j < my_n_stop; j++)
            whole_image.image_data[i][j] = u_bar.image_data[i][j];

        convert_image_to_jpeg(&whole_image, image_chars);
        export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
        deallocate_image(&whole_image);
    }

    MPI_Finalize();
    return 0;
}
