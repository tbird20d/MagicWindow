import cv2
import dlib
import numpy as np

ESC = 27

# Initialize dlib's face detector and load the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize nose tip reference position
reference_nose_tip_position = None

# Initialize the reference distance between eyes
reference_eye_distance = None

# def find_largest_contour(contours):
#     max_area = 0
#     largest_contour = None
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > max_area:
#             max_area = area
#             largest_contour = contour
#     return largest_contour

# Function to calculate distance between two points
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blank_frame = np.zeros_like(frame)

    # Detect faces
    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)

        landmark_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)])

        # Draw each landmark with lines spanning regions
        last_point = None
        for i, landmark in enumerate(landmark_points):
            cv2.circle(blank_frame, landmark, 1, (255, 0, 0), -1)
            if last_point is not None:
                cv2.line(blank_frame, last_point, landmark, (0, 255, 0), 1)
            last_point = landmark

            # Region separation indicies
            if i in [16, 21, 26, 35, 41, 47]:
                last_point = None
        
        # Get the nose tip position
        nose_tip_position = (landmarks.part(30).x, landmarks.part(30).y)
        
        # Define the left and right eye regions
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)], np.int32)
        for landmark in left_eye:
            cv2.circle(blank_frame, landmark, 2, (255, 0, 255), -1)  # Magenta dot for the eye landmarks
            
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)], np.int32)
        for landmark in right_eye:
            cv2.circle(blank_frame, landmark, 2, (255, 0, 255), -1)  # Magenta dot for the eye landmarks

        # Calculate the bounding box for each eye
        left_eye_bounds = cv2.boundingRect(left_eye)
        right_eye_bounds = cv2.boundingRect(right_eye)

        # Calculate the center of each bounding box
        left_eye_center = (left_eye_bounds[0] + left_eye_bounds[2] // 2, left_eye_bounds[1] + left_eye_bounds[3] // 2)
        right_eye_center = (right_eye_bounds[0] + right_eye_bounds[2] // 2, right_eye_bounds[1] + right_eye_bounds[3] // 2)

        # If this is the first frame with a detected face, or you want to reset the reference position periodically
        if reference_nose_tip_position is None:
            reference_nose_tip_position = nose_tip_position

        # Points for the left and right eye
        left_eye_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye_point = (landmarks.part(45).x, landmarks.part(45).y)

        # Calculate the distance between the eyes
        current_eye_distance = euclidean_dist(np.array(left_eye_point), np.array(right_eye_point))

        if reference_eye_distance is None:
            reference_eye_distance = current_eye_distance

        # Calculate depth change
        depth_change = current_eye_distance - reference_eye_distance
        # print("Depth change:", depth_change)

        # Calculate the movement from the reference position
        movement_vector = (nose_tip_position[0] - reference_nose_tip_position[0],
                           nose_tip_position[1] - reference_nose_tip_position[1],
                           int(depth_change))
        
        # Use movement_vector to determine the lateral movement
        # For simplicity, this example just prints the vector
        print("Movement vector:", movement_vector)

        # Visual feedback for the landmarks
        cv2.circle(blank_frame, nose_tip_position, 2, (0, 0, 255), -1)  # Red dot for the nose tip
        cv2.circle(blank_frame, left_eye_center, 2, (255, 255, 0), -1)  # Cyan dot for the left eye
        cv2.circle(blank_frame, right_eye_center, 2, (255, 255, 0), -1)  # Cyan dot for the right eye

    cv2.imshow("Frame", blank_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ESC:
        break

cap.release()
cv2.destroyAllWindows()
