from flask import Flask, redirect, render_template, request, Response, url_for
from PIL import Image
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import pymongo

app = Flask(__name__)

# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://FDR:FDRSYSTEM@fdr.txwnedp.mongodb.net/?retryWrites=true&w=majority")
db = client["FDR"]
# Initialize face detection and face recognition models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)



@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/')
def index():
    # List existing collections
    existing_collections = db.list_collection_names()
    return render_template('index.html', collections=existing_collections)





@app.route('/upload', methods=['POST'])
def upload():
    if 'file_1' not in request.files:
        return "No file part"

    num_people = int(request.form['num_people'])

    for i in range(1, num_people + 1):
        file = request.files.get(f'file_{i}')
        name = request.form.get(f'name_{i}')

        if file.filename == '':
            return f"No selected file for Person {i}"

        if file:
            # Load image
            img = Image.open(file)

            # Get other data from form
            collection_option = request.form['collection-option']

            if collection_option == 'create':
                # If creating a new collection, get the collection name from the form
                collection_name = request.form['collection']
                collection = db[collection_name]
            elif collection_option == 'existing':
                # If using an existing collection, get the collection name from the form
                collection_name = request.form['existing-collection']
                collection = db[collection_name]
            else:
                # Handle invalid collection option
                return "Invalid collection option"

            # Detect faces and extract embeddings
            faces, _ = mtcnn(img, return_prob=True)

            if faces is not None:
                # Get embeddings
                embeddings = resnet(faces).detach().cpu().numpy()

                # Flatten embeddings
                flattened_embeddings = embeddings.flatten().tolist()

                # Insert data into MongoDB
                new_document = {
                    "name": name,
                    "image_path": file.filename,
                    "vector_embedding": flattened_embeddings
                }
                collection.insert_one(new_document)
            else:
                return f"No faces detected in image for Person {i}"
        else:
            return f"Upload failed for Person {i}"

    return redirect(url_for('index'))  # Redirect to the index page after successful upload



def recognize_face(reference_data, query_embedding, threshold=0.7):
    for reference_embedding, name in reference_data:
        # Resize reference embedding to match query embedding dimensionality
        reference_embedding_resized = np.resize(reference_embedding, query_embedding.shape)
        similarity = 1 - cosine(reference_embedding_resized, query_embedding)
        if similarity >= threshold:
            return True, name, similarity
    return False, None, None

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save video to a temporary file
        video_path = 'temp_video.mp4'
        file.save(video_path)

        # Process the video
        result = process_video_file(video_path)

        if result == "interrupted":
            return "Video processing interrupted"

        return "Processing complete"

def process_video_file(video_path):
    # Open the selected collection
    collection_name = request.form['collection']
    collection = db[collection_name]

    # Retrieve reference embeddings and names from MongoDB
    reference_data = [(face["vector_embedding"], face["name"]) for face in collection.find({}, {"vector_embedding": 1, "name": 1})]

    # Define a distance threshold for good recognition
    distance_threshold = 0.5

    # Initialize counters
    known_counter = 0
    unknown_counter = 0

    # Maintain a list of recognized embeddings and their corresponding names
    recognized_embeddings = []
    recognized_unknown_embeddings = []

    # Main loop for processing video frames
    cap = cv2.VideoCapture(video_path)

    # Frame skip interval
    frame_skip = 7 
    frame_count = 0

    while cap.isOpened():
        # Capture frame from video
        ret, frame = cap.read()
        frame_count += 1

        # Skip frames if necessary
        if frame_count % frame_skip != 0:
            continue

        if not ret:
            break

        # Detect faces in frame
        boxes, _ = mtcnn.detect(frame)

        # If faces are detected, process each one
        if boxes is not None:
            for box in boxes:
                # Extract face from frame
                x1, y1, x2, y2 = [int(coord) for coord in box]
                face = frame[y1:y2, x1:x2]

                # Check if the face extraction was successful
                if face.size == 0:
                    continue  # Skip to the next box if face extraction failed

                # Preprocess query image
                query_image = cv2.resize(face, (160, 160))
                query_image = query_image / 255.0
                query_embedding = resnet(torch.tensor(query_image, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2).to(device)).squeeze().detach().cpu().numpy()

                # Recognize face with distance threshold
                recognized, name, similarity = recognize_face(reference_data, query_embedding, threshold=distance_threshold)

                # Increment counters based on recognition result
                if recognized:
                    if name not in recognized_embeddings:
                        known_counter += 1
                        recognized_embeddings.append(name)
                        print(f"Recognized: {name}")
                        # Remove the recognized face from recognized_unknown_embeddings if it was previously unknown
                        recognized_unknown_embeddings = [(known_embedding, sim) for known_embedding, sim in recognized_unknown_embeddings if cosine(known_embedding, query_embedding) >= distance_threshold]
                else:
                    # Check if the face is new or similar to an existing unknown face
                    is_new_face = True
                    for known_embedding, _ in recognized_unknown_embeddings:
                        if cosine(known_embedding, query_embedding) < distance_threshold:
                            is_new_face = False
                            break
                    if is_new_face:
                        unknown_counter += 1
                        recognized_unknown_embeddings.append((query_embedding, similarity))

                # Draw bounding box around face
                color = (0, 255, 0) if recognized else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name if recognized else "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Get frame dimensions
                frame_height, frame_width, _ = frame.shape

                # Define text to display
                text = f"Known: {known_counter} | Unknown: {unknown_counter}"

                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)

                # Calculate position to display text in the middle bottom
                text_x = int((frame_width - text_width) / 2)
                text_y = frame_height - 10  # Offset from the bottom

                # Draw black background rectangle for the text
                cv2.rectangle(frame, (text_x - 5, text_y - text_height - baseline - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)

                # Draw white text on black background with minimized font size
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # Display frame with smaller size
                cv2.imshow('Face Recognition', cv2.resize(frame, (int(frame_width / 2), int(frame_height / 2))))

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n--- Final Review ---")
            print(f"Total Known Faces Recognized: {known_counter}")
            print(f"Total Unknown Faces Detected: {unknown_counter}")
            return "interrupted"  # Return interrupted status if 'q' is pressed

    # Clear recognized embeddings for privacy
    recognized_embeddings = []
    recognized_unknown_embeddings = []

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return "complete"  # Return complete status if processing completes without interruption

@app.route('/start_stream', methods=['POST'])
def start_stream():
    collection_name = request.form['collection']
    collection = db[collection_name]
    reference_data = [(face["vector_embedding"], face["name"]) for face in collection.find({}, {"vector_embedding": 1, "name": 1})]
    distance_threshold = 0.5

    def gen_frames():
        cap = cv2.VideoCapture(0)
        known_counter = 0
        unknown_counter = 0
        Known_Person = []
        Unkown_Person = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    
                    query_image = cv2.resize(face, (160, 160)) / 255.0
                    query_embedding = resnet(torch.tensor(query_image, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2).to(device)).squeeze().detach().cpu().numpy()

                    recognized, name, similarity = recognize_face(reference_data, query_embedding, threshold=distance_threshold)

                    if recognized:
                        if name not in Known_Person:
                            known_counter += 1
                            Known_Person.append(name)
                            print(f"Recognized: {name}")  # Print recognized person's name
                            Unkown_Person = [(known_embedding, sim) for known_embedding, sim in Unkown_Person if cosine(known_embedding, query_embedding) >= distance_threshold]
                    else:
                        is_new_face = True
                        for known_embedding, _ in Unkown_Person:
                            if cosine(known_embedding, query_embedding) < distance_threshold:
                                is_new_face = False
                                break
                        if is_new_face:
                            unknown_counter += 1
                            Unkown_Person.append((query_embedding, similarity))

                    color = (0, 255, 0) if recognized else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name if recognized else "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            frame_height, frame_width, _ = frame.shape

            text = f"Known: {known_counter} | Unknown: {unknown_counter}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)
            text_x = int((frame_width - text_width) / 2)
            text_y = frame_height - 10

            cv2.rectangle(frame, (text_x - 5, text_y - text_height - baseline - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
