
# Face Detection and Blurring with OpenCV and Mediapipe

This project uses OpenCV and Mediapipe to detect faces in images, videos, or live webcam feed and applies a blur to the detected face regions for privacy protection.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Image Mode](#image-mode)
  - [Video Mode](#video-mode)
  - [Webcam Mode](#webcam-mode)
- [Code Explanation](#code-explanation)
- [Future Improvements](#future-improvements)

## Overview

The project provides real-time face detection and blurring functionality. It can process images, video files, or a live webcam feed. The blurring effect is applied only to detected face regions, ensuring the rest of the frame remains unaffected.

## Features

- **Multiple modes**: Supports image, video, and webcam modes for face detection and blurring.
- **Face detection**: Uses Mediapipe's face detection module for accurate face tracking.
- **Real-time processing**: Processes video and webcam streams in real-time.
- **Configurable**: Simple command-line arguments allow you to switch between modes and specify file paths.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Mediapipe (`mediapipe`)

You can install the required packages by running:
```bash
pip install opencv-python mediapipe
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/face-detection-blur.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd face-detection-blur
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The script can be run in three modes: `image`, `video`, and `webcam`.

### Image Mode
To process a single image and blur the faces:
```bash
python main.py --mode image --filePath path/to/image.jpg
```
The processed image with blurred faces will be saved to the `./output` directory.

### Video Mode
To process a video file and blur the faces:
```bash
python main.py --mode video --filePath path/to/video.mp4
```
The processed video will be saved to the `./output` directory as `output.mp4`.

### Webcam Mode
To use the live webcam feed for real-time face detection and blurring:
```bash
python main.py --mode webcam
```
Press `q` to exit the webcam mode.

## Code Explanation

### `main.py`
- The script uses the `argparse` module to handle command-line arguments, enabling the user to choose between image, video, and webcam modes.
- **Image mode**: Loads an image, detects faces, applies a blur to the face regions, and saves the output.
- **Video mode**: Reads a video file frame by frame, detects and blurs faces, and saves the processed video.
- **Webcam mode**: Continuously captures video from the webcam, detects and blurs faces in real-time, and displays the processed video feed.

### `process_img()`
- This function processes the input image (or video frame), detects faces using Mediapipe’s face detection, and applies a blur to the face bounding box region.

## Future Improvements

- **Multiple face models**: Add support for other face detection models.
- **Adjustable blur strength**: Allow the user to configure the blur intensity.
- **Additional effects**: Implement other privacy effects like pixelation.