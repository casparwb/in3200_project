

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

void iso_diffusion_denoising(image *u, image*u_bar, float kappa, int iters)
{
  // carrying out iters iterations of isotropic diffusion on a noisy image object u
  // denoised image should be stored and returned in the u_bar object

  int i, j, k, m = (*u).m, n = (*u).n;
  float **temp_ptr;

  for (i = 0; i < m; i++){
    (*u_bar).image_data[i][0] = (*u).image_data[i][0];
    (*u_bar).image_data[i][n-1] = (*u).image_data[i][n-1];
  }

  for (i = 1; i < n; i++){
    (*u_bar).image_data[0][i] = (*u).image_data[0][i];
    (*u_bar).image_data[m-1][i] = (*u).image_data[m-1][i];
  }

  for (k = 0; k < iters; k++){
    for (i = 1; i < m-1; i++)
      for (j = 1; j < n-1; j++)
        (*u_bar).image_data[i][j] = (*u).image_data[i][j] +
                                    kappa*
                                    ((*u).image_data[i-1][j] +
                                    (*u).image_data[i][j-1]-
                                    4*(*u).image_data[i][j] +
                                    (*u).image_data[i][j+1] +
                                    (*u).image_data[i+1][j]);


    temp_ptr = (*u_bar).image_data;
    (*u_bar).image_data = (*u).image_data;
    (*u).image_data = temp_ptr;
  }

}
