/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "ros_compat.h"
#include "image_converter.h"
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <jetson-inference/detectNet.h>

#include <unordered_map>

// Define ExactTime sync policy
typedef message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image,sensor_msgs::msg::Image> ExactSyncPolicy;

// globals
detectNet* net = NULL;
uint32_t overlay_flags = detectNet::OVERLAY_NONE;

imageConverter* input_cvt   = NULL;
imageConverter* overlay_cvt = NULL;

Publisher<vision_msgs::Detection2DArray> detection_pub = NULL;
Publisher<sensor_msgs::Image> overlay_pub = NULL;
Publisher<vision_msgs::VisionInfo> info_pub = NULL;

vision_msgs::VisionInfo info_msg;


// triggered when a new subscriber connected
void info_callback()
{
	ROS_INFO("new subscriber connected to vision_info topic, sending VisionInfo msg");
	info_pub->publish(info_msg);
}

// publish overlay create overlay image using opencv
bool publish_overlay(const sensor_msgs::msg::Image::ConstSharedPtr &input,
					 const cv_bridge::CvImagePtr &cv_ptr, 
					 detectNet::Detection* detections, int numDetections)
{
	// draw bounding boxes, labels, and confidence on the image
	for (int i = 0; i < numDetections; i++)
	{
		detectNet::Detection* det = detections + i;
		cv::Rect bbox(det->Left, det->Top, det->Width(), det->Height());
		cv::rectangle(cv_ptr->image, bbox, cv::Scalar(0, 255, 0), 2);

		std::string label = net->GetClassDesc(det->ClassID);
		std::stringstream ss;
		ss << label << " (" << det->Confidence << "," << det->MeanDistance << ")";
		cv::putText(cv_ptr->image, ss.str(), cv::Point(det->Left, det->Top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
	}

	// convert the image back to ROS format
	// Convert back to ROS and publish
	cv_bridge::CvImage out_msg;
    out_msg.header = input->header; // Same timestamp and tf frame as input image
    out_msg.encoding = sensor_msgs::image_encodings::BGR8; // Or whatever
    out_msg.image = cv_ptr->image;

	// publish the overlay image
	overlay_pub->publish(*(out_msg.toImageMsg()));

	return true;
}

// publish overlay image
bool publish_overlay( detectNet::Detection* detections, int numDetections )
{
	// get the image dimensions
	const uint32_t width  = input_cvt->GetWidth();
	const uint32_t height = input_cvt->GetHeight();

	// assure correct image size
	if( !overlay_cvt->Resize(width, height, imageConverter::ROSOutputFormat) )
		return false;

	ROS_INFO("overlay %ux%u image", width, height);
	// generate the overlay
	if( !net->Overlay(input_cvt->ImageGPU(), overlay_cvt->ImageGPU(), width, height, 
				   imageConverter::InternalFormat, detections, numDetections, overlay_flags) )
	{
		return false;
	}
	ROS_INFO("overlay generated");
	// populate the message
	sensor_msgs::Image msg;

	if( !overlay_cvt->Convert(msg, imageConverter::ROSOutputFormat) )
		return false;

	// populate timestamp in header field
	msg.header.stamp = ROS_TIME_NOW();

	// publish the message	
	overlay_pub->publish(msg);
	ROS_DEBUG("publishing %ux%u overlay image", width, height);
	return true;
}


// input image subscriber callback
void img_callback( const sensor_msgs::msg::Image::ConstSharedPtr &input_color, 
				   const sensor_msgs::msg::Image::ConstSharedPtr &input_depth )
{
	// convert the image to reside on GPU
	if( !input_cvt || !input_cvt->Convert(input_color) )
	{
		ROS_INFO("failed to convert %ux%u %s image", input_color->width, input_color->height, input_color->encoding.c_str());
		return;	
	}

	// classify the image
	detectNet::Detection* detections = NULL;

	const int numDetections = net->Detect(input_cvt->ImageGPU(), input_cvt->GetWidth(), input_cvt->GetHeight(), &detections, detectNet::OVERLAY_NONE);

	// verify success	
	if( numDetections < 0 )
	{
		ROS_ERROR("failed to run object detection on %ux%u image", input_color->width, input_color->height);
		return;
	}

	// convert the input image to OpenCV format
	cv_bridge::CvImagePtr cv_ptr_color;
	cv_bridge::CvImageConstPtr cv_ptr_depth;
	try
	{
		cv_ptr_color = cv_bridge::toCvCopy(input_color, sensor_msgs::image_encodings::BGR8);
		cv_ptr_depth = cv_bridge::toCvShare(input_depth, sensor_msgs::image_encodings::TYPE_16UC1);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return ;
	}

	// if objects were detected, send out message
	if( numDetections > 0 )
	{
		ROS_INFO("detected %i objects in %ux%u image", numDetections, input_color->width, input_color->height);
		
		// create a detection for each bounding box
		vision_msgs::Detection2DArray msg;

		for( int n=0; n < numDetections; n++ )
		{
			detectNet::Detection* det = detections + n;

			ROS_INFO("object %i class #%u (%s)  confidence=%f", n, det->ClassID, net->GetClassDesc(det->ClassID), det->Confidence);
			ROS_INFO("object %i bounding box (%f, %f)  (%f, %f)  w=%f  h=%f", n, det->Left, det->Top, det->Right, det->Bottom, det->Width(), det->Height()); 
			cv::Rect roi(det->Left, det->Top, det->Width(), det->Height());
			// extract roi from the depth image
			cv::Mat depth_roi = cv_ptr_depth->image(roi);
			// get the mean depth value
			cv::Scalar mean_depth = cv::mean(depth_roi);
			ROS_INFO("object %i mean depth value = %f", n, mean_depth[0]);
			det->MeanDistance = mean_depth[0];
			// create a detection sub-message
			vision_msgs::Detection2D detMsg;

			detMsg.bbox.size_x = det->Width();
			detMsg.bbox.size_y = det->Height();
			
			float cx, cy;
			det->Center(&cx, &cy);

		#if ROS_DISTRO >= ROS_HUMBLE
			detMsg.bbox.center.position.x = cx;
			detMsg.bbox.center.position.y = cy;
		#else
			detMsg.bbox.center.x = cx;
			detMsg.bbox.center.y = cy;
		#endif
		
			detMsg.bbox.center.theta = 0.0f;		// TODO optionally output object image

			// create classification hypothesis
			vision_msgs::ObjectHypothesisWithPose hyp;
			
		#if ROS_DISTRO >= ROS_GALACTIC
			hyp.hypothesis.class_id = det->ClassID;
			hyp.hypothesis.score = det->Confidence;
		#else
			hyp.id = det->ClassID;
			hyp.score = det->Confidence;
		#endif
		
			detMsg.results.push_back(hyp);
			msg.detections.push_back(detMsg);
		}

		// populate timestamp in header field
		msg.header.stamp = ROS_TIME_NOW();

		// publish the detection message
		detection_pub->publish(msg);
	} else {
		ROS_DEBUG("no objects detected in %ux%u image", input_color->width, input_color->height);
	}

	// generate the overlay (if there are subscribers)
	if( ROS_NUM_SUBSCRIBERS(overlay_pub) > 0 )
		publish_overlay(input_color,cv_ptr_color,detections, numDetections);
}


// node main loop
int main(int argc, char **argv)
{
	/*
	 * create node instance
	 */
	ROS_CREATE_NODE("detectnet");

	/*
	 * retrieve parameters
	 */	
	std::string model_name  = "ssd-mobilenet-v2";
	std::string model_path;
	std::string prototxt_path;
	std::string class_labels_path;
	
	std::string input_blob  = DETECTNET_DEFAULT_INPUT;
	std::string output_cvg  = DETECTNET_DEFAULT_COVERAGE;
	std::string output_bbox = DETECTNET_DEFAULT_BBOX;
	std::string overlay_str = "box,labels,conf";

	float mean_pixel = 0.0f;
	float threshold  = DETECTNET_DEFAULT_THRESHOLD;

	ROS_DECLARE_PARAMETER("model_name", model_name);
	ROS_DECLARE_PARAMETER("model_path", model_path);
	ROS_DECLARE_PARAMETER("prototxt_path", prototxt_path);
	ROS_DECLARE_PARAMETER("class_labels_path", class_labels_path);
	ROS_DECLARE_PARAMETER("input_blob", input_blob);
	ROS_DECLARE_PARAMETER("output_cvg", output_cvg);
	ROS_DECLARE_PARAMETER("output_bbox", output_bbox);
	ROS_DECLARE_PARAMETER("overlay_flags", overlay_str);
	ROS_DECLARE_PARAMETER("mean_pixel_value", mean_pixel);
	ROS_DECLARE_PARAMETER("threshold", threshold);


	/*
	 * retrieve parameters
	 */
	ROS_GET_PARAMETER("model_name", model_name);
	ROS_GET_PARAMETER("model_path", model_path);
	ROS_GET_PARAMETER("prototxt_path", prototxt_path);
	ROS_GET_PARAMETER("class_labels_path", class_labels_path);
	ROS_GET_PARAMETER("input_blob", input_blob);
	ROS_GET_PARAMETER("output_cvg", output_cvg);
	ROS_GET_PARAMETER("output_bbox", output_bbox);
	ROS_GET_PARAMETER("overlay_flags", overlay_str);
	ROS_GET_PARAMETER("mean_pixel_value", mean_pixel);
	ROS_GET_PARAMETER("threshold", threshold);

	overlay_flags = detectNet::OverlayFlagsFromStr(overlay_str.c_str());


	/*
	 * load object detection network
	 */
	if( model_path.size() > 0 )
	{
		// create network using custom model paths
		net = detectNet::Create(prototxt_path.c_str(), model_path.c_str(), 
						    mean_pixel, class_labels_path.c_str(), threshold, 
						    input_blob.c_str(), output_cvg.c_str(), output_bbox.c_str());
	}
	else
	{
		// create network using the built-in model
		net = detectNet::Create(model_name.c_str());
	}

	if( !net )
	{
		ROS_ERROR("failed to load detectNet model");
		return 0;
	}


	/*
	 * create the class labels parameter vector
	 */
	std::hash<std::string> model_hasher;  // hash the model path to avoid collisions on the param server
	std::string model_hash_str = std::string(net->GetModelPath()) + std::string(net->GetClassPath());
	const size_t model_hash = model_hasher(model_hash_str);
	
	ROS_INFO("model hash => %zu", model_hash);
	ROS_INFO("hash string => %s", model_hash_str.c_str());

	// obtain the list of class descriptions
	std::vector<std::string> class_descriptions;
	const uint32_t num_classes = net->GetNumClasses();

	for( uint32_t n=0; n < num_classes; n++ )
		class_descriptions.push_back(net->GetClassDesc(n));

	// create the key on the param server
	std::string class_key = std::string("class_labels_") + std::to_string(model_hash);

	ROS_DECLARE_PARAMETER(class_key, class_descriptions);
	ROS_SET_PARAMETER(class_key, class_descriptions);
		
	// populate the vision info msg
	std::string node_namespace = ROS_GET_NAMESPACE();
	ROS_INFO("node namespace => %s", node_namespace.c_str());

	info_msg.database_location = node_namespace + std::string("/") + class_key;
	info_msg.database_version  = 0;
	info_msg.method 		  = net->GetModelPath();
	
	ROS_INFO("class labels => %s", info_msg.database_location.c_str());


	/*
	 * create image converter objects
	 */
	input_cvt = new imageConverter();
	overlay_cvt = new imageConverter();

	if( !input_cvt || !overlay_cvt )
	{
		ROS_ERROR("failed to create imageConverter objects");
		return 0;
	}


	/*
	 * advertise publisher topics
	 */
	ROS_CREATE_PUBLISHER(vision_msgs::Detection2DArray, "detections", 25, detection_pub);
	ROS_CREATE_PUBLISHER(sensor_msgs::Image, "overlay", 2, overlay_pub);
	
	ROS_CREATE_PUBLISHER_STATUS(vision_msgs::VisionInfo, "vision_info", 1, info_callback, info_pub);


	/*
	 * subscribe to image topic
	 */
	auto image_color_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(node, "image_in_color");
	auto image_depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(node, "image_in_depth");

    // Create the synchronizer with the policy
    auto sync_ = std::make_shared<message_filters::Synchronizer<ExactSyncPolicy>>(ExactSyncPolicy(10), *image_color_sub_, *image_depth_sub_);
    sync_->registerCallback(std::bind(&img_callback, std::placeholders::_1, std::placeholders::_2));
   
	//auto img_sub = ROS_CREATE_SUBSCRIBER(sensor_msgs::Image, "image_in", 5, img_callback);
	
	/*
	 * wait for messages
	 */
	ROS_INFO("detectnet node initialized, waiting for messages");
	ROS_SPIN();


	/*
	 * free resources
	 */
	delete net;
	delete input_cvt;
	delete overlay_cvt;

	return 0;
}

