from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np  # Import numpy for numerical operations
import os

app = Flask(__name__)

# Initialize global variables
camera_index = 0  # Default camera index
camera = None
is_paused = False  # Track pause/resume state

# List of all defect types
defect_types = [
    "holes", "foreign_yarn", "surface_contamination", "slubs",
    "dirty_mark", "knots", "oil_mark", "miss_print", "dye_patch"
]

# Create folders for all defect types
for defect in defect_types:
    os.makedirs(f'data/{defect}', exist_ok=True)

def initialize_camera():
    """Initialize the camera."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            print(f"Error: Could not open camera at index {camera_index}")
            camera = None

def release_camera():
    """Release the camera."""
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None

def preprocess_image(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame to 64x64
    resized_frame = cv2.resize(gray_frame, (64, 64))
    
    # Normalize the frame
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    
    # Expand dimensions to add channel for grayscale (1 channel)
    expanded_frame = np.expand_dims(normalized_frame, axis=-1)
    
    # Convert grayscale to 3 channels by duplicating the grayscale channel
    rgb_frame = np.concatenate([expanded_frame] * 3, axis=-1)
    
    # Convert to 8-bit unsigned integer (0-255)
    rgb_frame = (rgb_frame * 255).astype(np.uint8)
    
    # Add batch dimension (if needed by the model)
    batch_frame = np.expand_dims(rgb_frame, axis=0)
    
    return batch_frame

def generate_frames():
    """Generate video frames for the video feed."""
    global is_paused, camera
    while True:
        if not camera or not camera.isOpened():
            initialize_camera()  # Try to reinitialize the camera
            if not camera or not camera.isOpened():
                print("Error: Camera is not available")
                break

        if is_paused:
            continue  # Skip frame grabbing when paused

        success, frame = camera.read()
        if not success:
            print("Error: Failed to grab frame from camera")
            break

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control_camera():
    """Handle camera controls: start, pause/resume, stop."""
    global is_paused, camera
    action = request.form.get('action')
    if action == "pause":
        is_paused = True
        return jsonify({'status': 'paused'})
    elif action == "resume":
        is_paused = False
        return jsonify({'status': 'resumed'})
    elif action == "stop":
        release_camera()
        return jsonify({'status': 'stopped'})
    elif action == "start":
        initialize_camera()
        if camera and camera.isOpened():
            return jsonify({'status': 'started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start camera'}), 500
    return '', 400

@app.route('/capture', methods=['POST'])
def capture_image():
    """Capture, preprocess, and save an image."""
    global camera
    defect_type = request.form.get('defect_type')
    if not camera or not camera.isOpened():
        return jsonify({'status': 'error', 'message': 'Camera is not active'}), 500

    success, frame = camera.read()
    if success and defect_type:
        # Preprocess the frame
        preprocessed_frame = preprocess_image(frame)
        # Remove batch dimension
        preprocessed_frame = np.squeeze(preprocessed_frame, axis=0)
        # Save the preprocessed image
        folder_path = f'data/{defect_type}'
        file_name = f"{folder_path}/{defect_type}_{len(os.listdir(folder_path)) + 1}.jpg"
        cv2.imwrite(file_name, preprocessed_frame)
        return jsonify({'status': 'image_saved', 'path': file_name})
    return jsonify({'status': 'error', 'message': 'Failed to capture image'}), 500

if __name__ == '__main__':
    initialize_camera()  # Ensure the camera is initialized before starting
    app.run(debug=True, threaded=True)  # Enable threading for better performance
