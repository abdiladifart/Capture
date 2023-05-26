import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Define the video capture object
cap = cv2.VideoCapture(0)

# Define the output directory
output_directory = "landmark_lines_frames"
os.makedirs(output_directory, exist_ok=True)

# Initialize the holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    frame_counter = 0
    while True:
        # Capture a frame from the video
        ret, frame = cap.read()

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = holistic.process(frame_rgb)

        # Create a black background
        black_frame = np.zeros_like(frame)

        # Draw only the lines of the landmarks on the black background
        mp_drawing.draw_landmarks(
            black_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            black_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            black_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Save the black frame with the lines as an image
        output_file = os.path.join(
            output_directory, f"frame_{frame_counter}.jpg")
        cv2.imwrite(output_file, black_frame)

        # Write part names and coordinates to the output file
        with open(os.path.join(output_directory, "/Users/frxan/Desktop/program/landmark_coordinates.txtt"), "a") as f:
            f.write(f"Frame {frame_counter}\n")
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    landmark_x = landmark.x * frame.shape[1]
                    landmark_y = landmark.y * frame.shape[0]
                    # Not all landmarks have z-coordinate
                    landmark_z = landmark.z * frame.shape[1]
                    f.write(
                        f"Pose Landmark: (x={landmark_x}, y={landmark_y}, z={landmark_z})\n")
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    landmark_x = landmark.x * frame.shape[1]
                    landmark_y = landmark.y * frame.shape[0]
                    # Not all landmarks have z-coordinate
                    landmark_z = landmark.z * frame.shape[1]
                    f.write(
                        f"Left Hand Landmark: (x={landmark_x}, y={landmark_y}, z={landmark_z})\n")
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    landmark_x = landmark.x * frame.shape[1]
                    landmark_y = landmark.y * frame.shape[0]
                    # Not all landmarks have z-coordinate
                    landmark_z = landmark.z * frame.shape[1]
                    f.write(
                        f"Right Hand Landmark: (x={landmark_x}, y={landmark_y}, z={landmark_z})\n")
            f.write("\n")

        # Display the frame
        cv2.imshow("frame", black_frame)

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1

# Release the video capture object and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Combine the saved frames into a video using FFmpeg
output_video = "landmark_lines.mp4"
os.system(
    f"ffmpeg -y -framerate 30 -i {output_directory}/frame_%d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p {output_video}")

# Remove the temporary image frames
for filename in os.listdir(output_directory):
    file_path = os.path.join(output_directory, filename)
    os.remove(file_path)

# Remove the output directory
os.rmdir(output_directory)

print(f"Video saved as {output_video}")
