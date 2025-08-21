import cv2
import face_recognition

# Open webcam
webcam = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    
    # Create a copy of the frame for the connected points
    frame_with_lines = frame.copy()

    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    for face_landmarks in face_landmarks_list:
        for feature, points in face_landmarks.items():
            for index, point in enumerate(points):
                # Draw the point index on the frame
                cv2.circle(frame , point , radius = 1 , color = (0 , 255 , 255), thickness =- 1)
                cv2.putText(frame , str(index) , point , cv2.FONT_HERSHEY_SIMPLEX , 0.4 , (0 , 255 , 255) , 1)
                
                # Draw lines connecting points on the copy
                if index > 0:  # Connect the current point to the previous one
                    cv2.line(frame_with_lines, points[index - 1], point, (0 , 255 , 255), 1)

            # Optionally close the loop for circular features
            if feature in ["top_lip", "bottom_lip", "left_eye", "right_eye"]:
                cv2.line(frame_with_lines, points[-1], points[0], (0 , 255 , 255), 1)

    # Display the frame with indices and connected points
    cv2.imshow("Facial Landmarks - Indices" , frame)
    cv2.imshow("Facial Landmarks - Connected Points", frame_with_lines)
    
    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


print(face_landmarks.keys())
# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()
