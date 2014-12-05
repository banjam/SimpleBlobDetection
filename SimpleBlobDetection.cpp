// SimpleBlobDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.
	if (!stream1.isOpened()){
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	int frame_width = stream1.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = stream1.get(CV_CAP_PROP_FRAME_HEIGHT);

	Mat frame, frame_gray, frame_thres, blob_image, window_image;
	int frameNumber = 1, fps = 10;
	int thresholdValueMin = 150, thresholdValueMax = 200;

	VideoWriter video("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, Size(frame_width * 2, frame_height), true);
	if (!video.isOpened())
	{
		cout << "Could not open the output video to write: " << endl;
		return -1;
	}

	string filefps = "fps.txt"; // framerate written to this file
	FILE* ffps;
	errno_t errfps = fopen_s(&ffps,filefps.c_str(), "w"); 

	string fileblobs = "blobs.txt"; // blobs info written to this file 
	FILE* fblobs;
	errno_t errblobs = fopen_s(&fblobs, fileblobs.c_str(), "w"); 

	double framerate_total = 0;

	//unconditional loop
	while (true) {
		int64 start = getTickCount();

		stream1 >> frame;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		blur(frame_gray, frame_gray, Size(3, 3));

		SimpleBlobDetector::Params params;
		params.thresholdStep = 25;
		params.minThreshold = thresholdValueMin;
		params.maxThreshold = thresholdValueMax;

		params.minDistBetweenBlobs = 2.0;  // minimum pixels between blobs
		params.filterByArea = false;
		//params.filterByArea = true;         // filter my blobs by area of blob
		//params.minArea = 2.0;              // min 2 pixels squared
		//params.maxArea = 500.0;             // max 500 pixels squared

		params.filterByColor = false;

		params.filterByCircularity = false;

		SimpleBlobDetector myBlobDetector(params);
		vector<KeyPoint> myBlobs;
		myBlobDetector.detect(frame_thres, myBlobs);

		//If you then want to have these keypoints highlighted on your image :

		threshold(frame_gray, frame_thres, thresholdValueMin, 255, 0);


		drawKeypoints(frame_thres, myBlobs, blob_image, Scalar(0, 0, 255));

		// Copy both frames into one image for display

		Mat window_image(frame_height, frame_width *2, CV_8UC3);
		Mat left(window_image, Rect(0, 0, frame_width, frame_height));
		frame.copyTo(left);
		Mat right(window_image, Rect(frame_width, 0, frame_width, frame_height));
		blob_image.copyTo(right);

		imshow("cam", window_image);
		moveWindow("cam", 0, 10);

		video.write(window_image);

		//// Write size and position of blobs to text file
		//fprintf(fblobs, "Frame %d \n", frameNumber);
		//fprintf(fblobs, "Blob x | Blob y | Blob size \n");
		//for (vector<KeyPoint>::iterator blobiterator = myBlobs.begin(); blobiterator != myBlobs.end(); blobiterator++)
		//{
		//	fprintf(fblobs, "%d \t", blobiterator->pt.x);
		//	fprintf(fblobs, "%d \t", blobiterator->pt.y);
		//	fprintf(fblobs, "%f \t", blobiterator->size);
		//	fprintf(fblobs, "\n");
		//}

		frameNumber++;

		if (waitKey(30) >= 0)
			break;
		framerate_total += getTickFrequency() / (getTickCount() - start);		
	}

	double framerate_average = framerate_total / frameNumber;
	cout << "FPS : " << framerate_average << endl;
	fprintf(ffps, "FPS: %f \n", framerate_average);
	fclose(ffps);
	fclose(fblobs);


	return 0;
}

