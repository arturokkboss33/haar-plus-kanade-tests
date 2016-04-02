#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <ctype.h>
#include <stdlib.h>
#include <string>

std::string disp_window = "display";

int main( int argc, char** argv ){

	/// Receive parameters about the template to track
	char *x, *y, *w, *h;
	int xpos = (int)strtol(argv[1],&x,10);
	int ypos = (int)strtol(argv[2],&y,10);
	int width = (int)strtol(argv[3],&w,10);
	int height = (int)strtol(argv[4],&h,10);
	std::cout << xpos << " "  << ypos << " " << width << " " << height << std::endl;

	///Read image and extract template
	std::cout << "Showing template" << std::endl;
	cv::Mat orig = cv::imread("examples/hole1/frame0003.jpg");
	cv::Mat orig_cp = orig.clone();
	cv::Rect patch_roi(xpos,ypos,width,height);
	cv::Mat img_patch = orig(patch_roi);
	cv::Point patch_center((int)(xpos+width/2),(int)(ypos+height/2));
	cv::ellipse(orig_cp, patch_center, cv::Size( width/2, height/2), 0, 0, 360, cv::Scalar( 255, 0, 0 ), 2, 8, 0);
	cv::imshow(disp_window,orig_cp);
	cv::waitKey();

	///Compute features to track
	std::cout << "Calculating features to track" << std::endl;
	cv::Mat orig_gray;
	cvtColor(orig, orig_gray, cv::COLOR_BGR2GRAY);
	const int MAX_CORNERS = 100;
	std::vector<cv::Point2f>corners[2];
	cv::Mat mask = cv::Mat::zeros(orig_gray.rows, orig_gray.cols, CV_8UC1);
	cv::Mat patch_mask = cv::Mat::ones(img_patch.rows, img_patch.cols, CV_8UC1);
	patch_mask.copyTo(mask(patch_roi));
	goodFeaturesToTrack(orig_gray,corners[0],MAX_CORNERS,0.01,10,mask,3,1,0.04);

	/// Set the neeed parameters to find the refined corners
  	cv::Size subPixWinSize = cv::Size( 5, 5 );
  	cv::Size zeroZone = cv::Size( -1, -1 );
  	cv::TermCriteria termcrit = cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 200, 0.003 );
  	cornerSubPix(orig_gray, corners[0], subPixWinSize, zeroZone, termcrit);
  	for(int i = 0; i < corners[0].size(); i++){
  		cv::circle(orig_cp, corners[0][i], 3, cv::Scalar(0,255,0), -1, 8);
  		//corners[1].push_back(corners[0][i]);
  	}
 	cv::imshow(disp_window,orig_cp);
	cv::waitKey(); 

	/// Calculate optical flow
	std::cout << "Calculating optical flow" << std::endl;
	std::vector<uchar> status;
    std::vector<float> err;
    cv::Size optFlowWinSize = cv::Size(31,31);
    cv::Mat next_frame = cv::imread("examples/hole1/frame0005.jpg");
    cv::Mat next_frame_cp = next_frame.clone();
    cv::Mat next_frame_gray;
    cvtColor(next_frame,next_frame_gray, cv::COLOR_BGR2GRAY);
    calcOpticalFlowPyrLK(orig_gray, next_frame_gray, corners[0], corners[1], status, err, optFlowWinSize, 5, termcrit, 
    	0, 0.01);
    for(int i = 0; i < corners[1].size(); i++){
    	if(!status[i]){
    		if(corners[1][i].x > 0 && corners[1][i].x < orig_gray.cols && corners[1][i].y > 0 && corners[1][i].y < orig_gray.rows ){
	    		std::cout << "good status " << corners[1][i].x << " " << corners[1][i].y << std::endl;
	    		cv::circle(next_frame_cp, corners[0][i], 5, cv::Scalar(0,255,0), -1, 8);
	  			cv::circle(next_frame_cp, corners[1][i], 3, cv::Scalar(255,0,255), -1, 8);
	  			cv::line(next_frame_cp, corners[0][i], corners[1][i], cv::Scalar(255, 0,0),1,8,0);
	  		}
    	}
  	}
  	std::cout << "Drawing" << std::endl;
  	cv::imshow(disp_window,next_frame_cp);
	cv::waitKey();



	return 0;

}
