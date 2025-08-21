import cv2
import numpy as np
import face_recognition

# Load data
glasses = cv2.imread("glasses.png" , cv2.IMREAD_UNCHANGED)
webcam = cv2.VideoCapture(0)

def overlay_image_alpha(background , overlay , x , y):
    """
    Overlay `overlay` image onto `background` at position (x, y).
    Handles transparency using alpha channel.
    """
    h, w = overlay.shape[ : 2]
    for i in range(h) :
        for j in range(w) :
            if x + j >= background.shape[1] or y + i >= background.shape[0] :
                continue
            if x + j < 0 or y + i < 0 :
                continue
            alpha = overlay[i , j , 3] / 255.0  # Alpha channel
            if alpha > 0:  # If pixel is not fully transparent
                background[y + i , x + j] = (alpha * overlay[i , j , : 3] + (1 - alpha) * background[y + i , x + j])


while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    
    # Detect face landmarks
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    
    for face_landmarks in face_landmarks_list:
        # Get key points for eyes
        left_eye = face_landmarks["left_eye"]
        right_eye = face_landmarks["right_eye"]

        # Calculate eye centers
        left_eye_center = np.mean(left_eye , axis = 0).astype(int)
        right_eye_center = np.mean(right_eye , axis = 0).astype(int)

        # Approximate ear positions using the jawline
        jawline = face_landmarks["chin"]
        left_ear = jawline[0]  # Leftmost point of jawline
        right_ear = jawline[-1]  # Rightmost point of jawline
        
        # Calculate glasses size and position
        glasses_width = int(np.linalg.norm(np.array(left_ear) - np.array(right_ear)))
        glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])

        # Scale up the glasses slightly
        scale_factor = 1.5  # Increase size
        glasses_width = int(glasses_width * scale_factor)
        glasses_height = int(glasses_height * scale_factor)
        resized_glasses = cv2.resize(glasses, (glasses_width, glasses_height))

        # Determine top-left corner of glasses
        x_offset = int(glasses_width * -0.1)  # Push to the right by 10% of the width
        top_left_x = left_eye_center[0] - glasses_width // 4 + x_offset
        top_left_y = left_eye_center[1] - glasses_height // 2

       
        # Overlay glasses
        overlay_image_alpha(frame, resized_glasses , top_left_x , top_left_y)

    # Show the frame
    cv2.imshow("Webcam" , frame)
                
    if cv2.waitKey(1) == ord("q") :
        break

webcam.release()
cv2.destroyAllWindows()