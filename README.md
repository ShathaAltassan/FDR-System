# <p align="center"> Face Detection Recognition </p>

<p align="center">
  <img src="https://github.com/ShathaAltassan/FDR-System/assets/138797663/09541351-1a96-47f8-a43c-9a90068dd1a8" alt="FDR Logo">
</p>

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
     
![image](https://github.com/ShathaAltassan/FDR-System/assets/138797663/93b37349-cf7c-48e3-b1db-0f5e21436875)

2. **Video Processing**:
   - Uploaded videos are processed frame by frame.
   - Each frame is analyzed using the MTCNN model to detect faces and the InceptionResnetV1 model to extract embeddings.
   - Recognition logic compares extracted embeddings with stored embeddings to identify known and unknown faces.
     
![image](https://github.com/ShathaAltassan/FDR-System/assets/138797663/999e19ca-1822-40b6-945d-a9011c5d6488)


3. **Live Streaming**:
   - The application supports live streaming from a camera source.
   - Live video feed undergoes real-time face detection and recognition, providing instantaneous recognition results.

## Database

MongoDB is used as the database for storing face embeddings and associated data. It enables efficient data management and retrieval, supporting the core functionality of the FDR System application.

- **Reliability**:
  - Throughout the database building process, the system demonstrates reliability by accurately saving uploaded data without errors or inconsistencies.

- **Stability**:
  - Stability is maintained even with varying amounts of data input, indicating robustness in handling different user requirements.

![image](https://github.com/ShathaAltassan/FDR-System/assets/138797663/6506fd71-c282-4287-90af-9ea26f1ff1b9)

