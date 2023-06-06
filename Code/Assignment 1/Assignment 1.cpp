#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

#define PRINT(x) std::cout << x << endl; // for debugging
using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.ppm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		CImg<unsigned char> convertedImage, yIntensity, cbChannel, crChannel; // for colour image
		bool colour = false;
		if (image_input.spectrum() == 3) {

			convertedImage = image_input.get_RGBtoYCbCr();
			yIntensity = convertedImage.get_channel(0);
			cbChannel = convertedImage.get_channel(1);
			crChannel = convertedImage.get_channel(2);
			colour = true;
		}

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image


		/// <summary>
		/// new stuff
		/// </summary>
		/// <param name="argc"></param>
		/// <param name="argv"></param>
		/// <returns></returns>

		typedef int mytype;

		//vector with range 256 for histogram
		std::vector<mytype> vec_Histogram(256);

		size_t histoSize = vec_Histogram.size() * sizeof(mytype);
		size_t input_elements = image_input.size(); // image_input defined on line 37

		std::vector<mytype> image_output(input_elements);
		size_t output_size = image_output.size() * sizeof(mytype);
		size_t input_size = image_input.size() * sizeof(mytype);
		size_t output_elements = image_output.size();
		size_t local_size = 10;
		int histBins = 256; // variable bin size

		// define buffers 
		// see line 76 and 77 for original image buffers: dev_image_input, dev-image_output
		cl::Buffer buffer_Hist(context, CL_MEM_READ_WRITE, histoSize);
		cl::Buffer buffer_cumHist(context, CL_MEM_READ_WRITE, histoSize);
		cl::Buffer buffer_normCH(context, CL_MEM_READ_WRITE, histoSize);
		cl::Buffer buffer_LUT(context, CL_MEM_READ_WRITE, image_input.size());

		//
		// basic histogram
		//
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(buffer_Hist, 0, 0, vec_Histogram.size());//zero B buffer on device memory

		// Global implementation of hist_simpleG						//		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		cl::Kernel kernel_1 = cl::Kernel(program, "hist_simpleG");
		kernel_1.setArg(0, dev_image_input);
		kernel_1.setArg(1, buffer_Hist);

		// local implementation of hist_simple
		//cl::Kernel kernel_1 = cl::Kernel(program, "hist_simple");
		//kernel_1.setArg(0, dev_image_input);
		//kernel_1.setArg(1, buffer_Hist);
		//kernel_1.setArg(2, cl::Local(histoSize));
		//kernel_1.setArg(3, histBins);

		// 
		//create event for runtime
		cl::Event prof_event1;

		//call kernel and read buffer for output
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event1);
		queue.enqueueReadBuffer(buffer_Hist, CL_TRUE, 0, histoSize, &vec_Histogram[0]);

		// output for basic histogram
		std::cout << "input_elements = " << input_elements << std::endl;
		std::cout << "Histogram = " << vec_Histogram << std::endl;
		// end of basic histogram
		//
		//
		// cumulative histogram

		std::vector<mytype> vec_cumHist(256);

		//write to and fill buffer with image with "histoSize" bins
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(buffer_cumHist, 0, 0, vec_cumHist.size()); // hist and cumhist use same number of bins.

		// setup kernels for execution
		cl::Kernel kernelCum = cl::Kernel(program, "hist_cum");
		kernelCum.setArg(0, buffer_Hist);
		kernelCum.setArg(1, buffer_cumHist);

		// create event for profiling info
		cl::Event prof_event2;

		//call kernel for cumulative histogram and read the output read for command line
		queue.enqueueNDRangeKernel(kernelCum, cl::NullRange, cl::NDRange(histoSize), cl::NullRange, NULL, &prof_event2);
		queue.enqueueReadBuffer(buffer_cumHist, CL_TRUE, 0, histoSize, &vec_cumHist[0]);

		//output for cumulative histogram
		std::cout << "Cumulative Histogram = " << vec_cumHist << std::endl;
		// end of cumulative histogram
		//
		// 
		// Normalised Cumulative Histogram

		std::vector<mytype> norm_cumHist(256);

		queue.enqueueWriteBuffer(buffer_normCH, CL_TRUE, 0, histoSize, &vec_cumHist.data()[0]);
		//queue.enqueueFillBuffer(buffer_normCH, 0, 0, norm_cumHist.size());

		 //opposite runtime, highlight and ctrl+k+u to uncomment, or ctrl+k+c to re-comment
		cl::Kernel kernelNormCH = cl::Kernel(program, "norm_cumHistG");
		kernelNormCH.setArg(0, buffer_cumHist);
		kernelNormCH.setArg(1, buffer_normCH);

		//cl::Kernel kernelNormCH = cl::Kernel(program, "norm_cumHist");
		//kernelNormCH.setArg(0, buffer_cumHist);
		//kernelNormCH.setArg(1, buffer_normCH);
		//kernelNormCH.setArg(2, cl::Local(vec_cumHist.size()));
		//kernelNormCH.setArg(3, cl::Local(norm_cumHist.size()));

		cl::Event prof_event3;

		queue.enqueueNDRangeKernel(kernelNormCH, cl::NullRange, cl::NDRange(histoSize), cl::NullRange, NULL, &prof_event3);
		queue.enqueueReadBuffer(buffer_normCH, CL_TRUE, 0, histoSize, &norm_cumHist[0]);

		std::cout << "Normalised Cumulative Histogram = " << norm_cumHist << std::endl;
		// end of Normalised Cumulative Histogram
		//
		//
		// Back Projection

		//queue.enqueueWriteBuffer(buffer_LUT, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);     lines arent needed but here for reference
		//queue.enqueueFillBuffer(buffer_LUT, 0, 0, image_input.size());

		cl::Kernel LUT = cl::Kernel(program, "lookupTable");
		LUT.setArg(0, dev_image_input);
		LUT.setArg(1, dev_image_output);
		LUT.setArg(2, buffer_normCH);

		//create event for lookupTable
		cl::Event prof_event4;

		queue.enqueueNDRangeKernel(LUT, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event4);
		queue.enqueueReadBuffer(buffer_LUT, CL_TRUE, 0, image_output.size(), &image_output[0]);    // needed for &prof_event to record in NDRangeKernel
		// end of LUT
		//
		//profiling info
		std::cout << "Kernel 1 execution time [ns]:" << prof_event1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event1.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Kernel 2 execution time [ns]:" << prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Kernel 3 execution time [ns]:" << prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Kernel 4 execution time [ns]:" << prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		//new image = RGB_toYcbcr (check syntax)
		//find spectrum and colour channel values.


		//
		// Prepare image with colour, also see lines 41 - 50
		vector<unsigned char> output_buffer(image_input.size());

		queue.enqueueReadBuffer(dev_image_input, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());

		if (colour == true)
		{
			for (int i = 0; i < output_image.width(); i++)
			{
				for (int j = 0; j < output_image.height(); j++)
				{
					convertedImage(i, j, 0, 0) = output_image(i, j);
					convertedImage(i, j, 1, 0) = cbChannel(i, j);
					convertedImage(i, j, 2, 0) = crChannel(i, j);
				}
			}
			output_image = convertedImage.get_YCbCrtoRGB();
		}

		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}