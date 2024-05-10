# Face Detection Recognition

## Overview

FDR System application designed for face recognition tasks. It leverages deep learning models for face detection and recognition and integrates MongoDB for efficient data storage. The application provides functionalities such as image upload, video processing, and live streaming for real-time face recognition.

## Key Features

- **Image Upload**: Users can upload images containing faces for recognition.
- **Video Processing**: The application processes uploaded videos frame by frame to detect and recognize faces.
- **Live Streaming**: Users can stream live video from a camera for real-time face recognition.
- **Database Integration**: MongoDB is used for storing face embeddings and associated data, enabling seamless data management and retrieval.

## Models Used

- **MTCNN (Multi-Task Cascaded Convolutional Networks)**: This model is utilized for face detection, providing bounding box coordinates for detected faces.
- **InceptionResnetV1**: Used for face recognition, extracting numerical embeddings of faces for comparison and recognition tasks.

## Functionality Overview

1. **Image Upload**:
   - Users can upload images through the application's user interface.
   - Uploaded images are processed using the MTCNN model to detect faces and extract embeddings.
   - Extracted embeddings are stored in MongoDB along with associated metadata (e.g., name or ID).

2. **Video Processing**:
   - Uploaded videos are processed frame by frame.
   - Each frame is analyzed using the MTCNN model to detect faces and the InceptionResnetV1 model to extract embeddings.
   - Recognition logic compares extracted embeddings with stored embeddings to identify known and unknown faces.

3. **Live Streaming**:
   - The application supports live streaming from a camera source.
   - Live video feed undergoes real-time face detection and recognition, providing instantaneous recognition results.


