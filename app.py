from flask import Flask, render_template, Response, request, jsonify
import cv2
import os

app = Flask(__name__)

# Initialize global variables
camera_index = 0  # Default camera index
camera = None
is_paused = False  # Track pause/resume state


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
    """Capture and save an image."""
    global camera
    defect_type = request.form.get('defect_type')
    if not camera or not camera.isOpened():
        return jsonify({'status': 'error', 'message': 'Camera is not active'}), 500

    success, frame = camera.read()
    if success and defect_type:
        folder_path = f'data/{defect_type}'
        file_name = f"{folder_path}/{defect_type}_{len(os.listdir(folder_path)) + 1}.jpg"
        cv2.imwrite(file_name, frame)
        return jsonify({'status': 'image_saved', 'path': file_name})
    return jsonify({'status': 'error', 'message': 'Failed to capture image'}), 500


if __name__ == '__main__':
    initialize_camera()  # Ensure the camera is initialized before starting
    app.run(debug=True, threaded=True)  # Enable threading for better performance
