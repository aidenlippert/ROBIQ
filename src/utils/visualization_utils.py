import cv2

def draw_pose_landmarks(frame, landmarks, connections, color=(0, 255, 0), radius=3):
    """
    Draw landmarks and connections on a video frame.
    Parameters:
    - frame: The video frame to draw on.
    - landmarks: List of landmarks to draw.
    - connections: List of tuples defining connections between landmarks.
    - color: Color for the landmarks and connections (BGR format).
    - radius: Radius for the landmark circles.
    Returns:
    - frame: The frame with landmarks and connections drawn.
    """
    for landmark in landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(frame, (x, y), radius, color, thickness=-1)

    for start_idx, end_idx in connections:
        start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
        end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
        cv2.line(frame, start_point, end_point, color, thickness=2)

    return frame