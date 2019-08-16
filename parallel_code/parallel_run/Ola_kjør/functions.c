#include <mpi.h>

typedef struct {
    float **image_data;  /* a 2D array of floats */
    int m;               /* # pixels in vertical-direction */
    int n;               /* # pixels in horizontal-direction */
} image;

void allocate_image(image *u, int m, int n) {
    (*u).image_data = (float**)malloc(m*sizeof(float*));

    for (int i = 0; i < m; i++) {
        (*u).image_data[i] = (float*)malloc(n*sizeof(float));

        for (int j = 0; j < n; j++) {
            (*u).image_data[i][j] = 0;
        }
    }
}

void deallocate_image(image *u) {
    for (int i = 0; i < (*u).m; i++) {
        free((*u).image_data[i]);
    }

    free((*u).image_data);
}

void convert_jpeg_to_image(const unsigned char* image_chars, image *u) {
    int m = (*u).m, n = (*u).n;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            (*u).image_data[i][j] = (float)(image_chars[i*n + j]);
        }
    }
}

void convert_image_to_jpeg(const image *u, unsigned char* image_chars) {
    int m = (*u).m, n = (*u).n;

    for (int i = 0; i < m; i ++) {
        for (int j = 0; j < n; j ++) {
        image_chars[i*n + j] = (unsigned char)(*u).image_data[i][j];
        }
    }
}

void pack(image *u, float *buf, int left) {
    /* left = 1 if left column is to be extracted */
    int m = (*u).m, n = (*u).n;

    if (left == 1) {
        for (int i = 1; i < m; i++) {
            buf[i] = (*u).image_data[i][1];
        }
    }
    else if (left == 0) {
        for (int i = 1; i < m; i++) {
            buf[i] = (*u).image_data[i][n - 2]; //sikre her?
        }
    }
}

void iso_diffusion_denoising_parallel(image *u, image *u_bar, float kappa,
                                      int  iters,MPI_Comm comm) {
    int my_m = (*u).m, my_n = (*u).n;
    float **temp;
    int my_rank, coord[2];

    MPI_Request request_x1, request_x2, request_y1, request_y2;
    MPI_Status status_x1, status_x2, status_y1, status_y2;

    MPI_Comm_rank (comm, &my_rank);
    MPI_Cart_coords(comm, my_rank, 2, coord);

    // mapping neighbouring procs and setting up buffers for ghost values
    int left, right, up, down;
    float *send_left, *send_right;
    float *receive_left, *receive_right, *receive_up, *receive_down;

    send_left = (float*)malloc(my_m*sizeof(float));
    send_right = (float*)malloc(my_m*sizeof(float));

    receive_left = (float*)malloc(my_m*sizeof(float));
    receive_right = (float*)malloc(my_m*sizeof(float));
    receive_up = (float*)malloc(my_m*sizeof(float));
    receive_down = (float*)malloc(my_m*sizeof(float));

    for (int k = 0; k < iters; k++) {

        // x-communication
        MPI_Cart_shift (comm, 1, 1, &left, &right);

        if (left != MPI_PROC_NULL) {
            MPI_Irecv(receive_left, my_m, MPI_FLOAT, left, 1, comm,&request_x1);
        }
        if (right != MPI_PROC_NULL) {
          MPI_Irecv(receive_right, my_m, MPI_FLOAT, right, 0, comm,
                    &request_x2);
        }
        if (right != MPI_PROC_NULL) {
            pack(u, send_right, 0);
            MPI_Send(send_right, my_m, MPI_FLOAT, right, 1, comm);
        }
        if (left != MPI_PROC_NULL) {
            pack(u, send_left, 1);
            MPI_Send(send_left, my_m, MPI_FLOAT, left, 0, comm);
        }
        if (left != MPI_PROC_NULL) {
            MPI_Wait(&request_x1, &status_x1);
        }
        if (right != MPI_PROC_NULL) {
            MPI_Wait(&request_x2, &status_x2);
        }

        // y-communication
        MPI_Cart_shift(comm, 0, 1, &up, &down);

        if (up != MPI_PROC_NULL) {
            MPI_Irecv(receive_up, my_n, MPI_FLOAT, up, 2, comm, &request_y1);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Irecv(receive_down, my_n, MPI_FLOAT, down, 3, comm,
                      &request_y2);
        }
        if (up != MPI_PROC_NULL) {
            MPI_Send((*u).image_data[1], my_n, MPI_FLOAT, up, 3, comm);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Send((*u).image_data[my_m - 2], my_n, MPI_FLOAT, down, 2, comm);
        }
        if (up != MPI_PROC_NULL) {
            MPI_Wait(&request_y1, &status_y1);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Wait(&request_y2, &status_y2);
        }

        for (int i = 1; i < my_m -1; i++) {
            for (int j = 1; j < my_n -1; j++) {
                (*u_bar).image_data[i][j] = (*u).image_data[i][j] + kappa*(
                                            (*u).image_data[i -1][j] +
                                            (*u).image_data[i][j -1] -
                                            4*(*u).image_data[i][j] +
                                            (*u).image_data[i][j +1] +
                                            (*u).image_data[i + 1][j]);
            }
        }

        if (my_rank == 0) {
            for (int i = 1; i < my_m - 1; i++) {
                (*u_bar).image_data[i][my_n] = (*u).image_data[i][my_n - 1] +
                                               kappa*(
                                               (*u).image_data[i -1][my_n - 1]+
                                               (*u).image_data[i][my_n - 1] -
                                               4*(*u).image_data[i][my_n - 1] +
                                               (*u).image_data[i][my_n + 1] +
                                               (*u).image_data[i + 1][my_n -1]);
            }
        }

        temp = (*u_bar).image_data;
        (*u_bar).image_data = (*u).image_data;
        (*u).image_data = temp;
    }
}
