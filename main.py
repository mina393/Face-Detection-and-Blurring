import cv2  # Import OpenCV for image and video processing
import mediapipe as mp  # Import Mediapipe for face detection
import os  # Import os for file and directory handling
import argparse  # Import argparse for command-line argument parsing

def process_img(img, face_detection):
    # Convert the image from BGR to RGB color space for face detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape  # Get the height and width of the image
    out = face_detection.process(img_rgb)  # Process the image to detect faces

    if out.detections is not None:  # If faces are detected
        for detection in out.detections:  # Loop through each detected face
            location_data = detection.location_data  # Get location data of the detected face
            bbox = location_data.relative_bounding_box  # Get the bounding box of the face

            # Calculate absolute bounding box coordinates
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Ensure bounding box coordinates are within valid limits
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            # Only blur the region if the coordinates are valid
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2, :] = cv2.blur(img[y1:y2, x1:x2, :], (40, 40))  # Apply blur to the detected face region

    return img  # Return the processed image

# Set up argument parser for command-line arguments
args = argparse.ArgumentParser()
args.add_argument("--mode", default="webcam")  # Set mode of operation (webcam, image, or video)
# args.add_argument("--filePath", default=r"D:\Modern\projects\opencv_project\data\face_detection_and_blurring\VideoBeforeBlurring.mp4")
args.add_argument("--filePath", default=None)  # Path for image or video file

args = args.parse_args()  # Parse the command-line arguments

output_dir = './output'  # Define output directory for saving results
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the output directory if it doesn't exist

mp_face_detection = mp.solutions.face_detection  # Initialize Mediapipe face detection

with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
    if args.mode in ["image"]:  # If mode is image
        img = cv2.imread(args.filePath)  # Read the input image
        if img is not None:  # Check if the image was loaded successfully
            img = process_img(img, face_detection)  # Process the image for face detection and blurring
            cv2.imwrite(os.path.join(output_dir, 'blured_image.jpg'), img)  # Save the processed image
        else:
            print("Error: Image not found or unable to read.")  # Error message if image fails to load
    elif args.mode == "video":  # If mode is video
        cap = cv2.VideoCapture(args.filePath)  # Open the video file
        if not cap.isOpened():
            print("Error: Could not open video.")  # Error message if video fails to open
            exit()

        ret, frame = cap.read()  # Read the first frame of the video
        if frame is None:
            print("Error: Could not read frame.")  # Error message if frame fails to read
            exit()

        # Set up video writer for saving processed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), fourcc, 25, (frame.shape[1], frame.shape[0]))

        while ret:  # Process video frame by frame
            processed_img = process_img(frame, face_detection)  # Process each frame
            output_video.write(processed_img)  # Write the processed frame to the output video

            ret, frame = cap.read()  # Read the next frame

        cap.release()  # Release the video capture object
        output_video.release()  # Release the video writer object
    elif args.mode in ['webcam']:  # If mode is webcam
        cap = cv2.VideoCapture(0)  # Open the default webcam
        ret, frame = cap.read()  # Read the first frame from the webcam

        while ret:  # Continuously process frames from the webcam
            img = process_img(frame, face_detection)  # Process the current frame
            cv2.imshow('frame', frame)  # Display the processed frame

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit loop if 'q' is pressed
            ret, frame = cap.read()  # Read the next frame

        cap.release()  # Release the webcam capture object
