from flask import Flask, render_template, Response
import cv2
import threading
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)
motion_count = 0  # Global variable to keep track of motion count
lock = threading.Lock()  # Lock to ensure thread safety for motion_count

@app.route('/app')
def index():
    return render_template('index.html', count=motion_count)

# Function for motion detection
def detect_motion():
    global motion_count  # Access the global variable
    
    camera = cv2.VideoCapture(0)
    _, prev_frame = camera.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_area = prev_frame.shape[0] * prev_frame.shape[1]  # Total number of pixels in the frame
    
    try:
        while True:
            _, frame = camera.read()
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(prev_frame, current_frame)
            _, frame_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            motion_pixels = cv2.countNonZero(frame_diff)
            motion_percentage = (motion_pixels / frame_area) * 100  # Calculate percentage of motion
            
            if motion_percentage > 1.0:  # Adjust this threshold as per your requirement
                with lock:
                    motion_count += 1  # Increment motion count
                if motion_count % 10 == 0:
                    print("Motion detected:", motion_count)
                    socketio.emit('motion_count', {'count': motion_count})
                
                # Check if motion count has reached 1000, then reset it to 0
                if motion_count >= 1000:
                    motion_count = 0
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')
                
            prev_frame = current_frame
            
    finally:
        camera.release()  # Release the camera when the loop ends

@app.route('/video_feed')
def video_feed():
    return Response(detect_motion(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start motion detection in a separate thread
    motion_thread = threading.Thread(target=detect_motion)
    motion_thread.daemon = True
    motion_thread.start()
     
    # Run the Flask app with SocketIO, allowing unsafe Werkzeug for production
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True,port=9876)
