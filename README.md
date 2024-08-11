# Vehicle Tracking and Speed Estimation

This project utilizes YOLOv8 and OpenCV to track vehicles and estimate their speed from video footage. It also counts the number of vehicles moving in both directions across a predefined line and displays their speeds.

## Features
Vehicle Tracking: Real-time tracking of vehicles using YOLOv8.

Speed Estimation: Estimates vehicle speeds.

Vehicle Counting: Counts vehicles moving in each direction across a specified line.

Visualization: Displays the video feed with tracked vehicles, speed information, and vehicle counts.

## Requirements

Python 3.8 or higher

Ultralytics YOLO

OpenCV

NumPy

To install the required libraries, use pip.

## Installation

1-Clone the repository:

    git clone https://github.com/yourusername/vehicle-tracking.git

    Navigate to the project directory:

    cd vehicle-tracking

2-Install the required libraries by running the requirements file.

3-Download the YOLOv8 pre-trained model from the Ultralytics YOLOv8 model repository and place it in the model directory.

## Usage

1-Place your video file in the project directory or update the file path in the script.

2-Run the script to start the vehicle tracking and speed estimation process.

3-The video feed will be displayed with real-time tracking and speed estimation. Press Esc to exit the video window.

## Code Explanation

YOLO Model Loading: Loads a pre-trained YOLO model for vehicle detection.
Speed Estimator Initialization: Initializes the speed estimator with the model classes and predefined line points.
Vehicle Tracking: Tracks vehicles across video frames and updates the counts based on their movement direction.
Drawing Functions: Includes functions to draw bounding boxes and text on the video frames.

## Contributing
Feel free to fork the repository, make changes, and submit pull requests. If you encounter any issues, please open an issue on GitHub.

