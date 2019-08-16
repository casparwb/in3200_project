The serial program is made with the attached makefile, which again uses the GCC compiler 

Running the produced output file requires the input of four arguments in the following sequence:
-------------------------------------------------------
1. a .jpg file in grey-scale
2. name of the output file
3. the kappa-value
4. number of iterations to run the iso-denoising function
-------------------------------------------------------

An example of how to compile and run the program:
-------------------------------------------------------
$ make

$ ./serial_main mona_lisa_noisy.jpg output_serial.jpg 0.2 100
--------------------------------------------------------

Which produces the output, including a denoised image file:
--------------------------------------------------------
Import successful! m = 4289 | n = 2835
Allocation successful!
Convertion to image successful!
Denoising successful!
Convertion to jpeg successful!
Deallocation successful!
Serial iso denoising time: 2s,  2234ms
---------------------------------------------------------

The parallel code is made with the attached makefile, which compiles the code with mpicc

The output file requires the same inputs as the serial code in the same order

An example of how to compile and run the program:
-------------------------------------------------------
$ make

$ mpirun -np 4 ./parallel_main mona_lisa_noisy.jpg output.jpg 0.2 100
--------------------------------------------------------


Which produces the output, including a denoised image file:
--------------------------------------------------------
Import successful! Image dimensions: 4289 x 2835
My rank: 1 | Iso parallel time: 1825.581789ms
My rank: 2 | Iso parallel time: 1776.438236ms
My rank: 3 | Iso parallel time: 1732.567072ms
My rank: 0 | Iso parallel time: 1730.857611ms
--------------------------------------
Global time taken by iso_denoising: 1825.581789s
---------------------------------------------------------