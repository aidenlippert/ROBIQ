import cv2

def list_available_cameras():
    """
    List all available cameras on the system.
    Returns:
    - cameras: A list of camera indices that are available.
    """
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        cameras.append(index)
        cap.release()
        index += 1
    return cameras

def set_camera_resolution(capture, width, height):
    """
    Set the resolution of the camera capture.
    Parameters:
    - capture: The cv2.VideoCapture object.
    - width: Desired width of the frame.
    - height: Desired height of the frame.
    """
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)