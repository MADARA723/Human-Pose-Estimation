import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import av

# --- CONFIGURATION ---
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth, inHeight = 368, 368

# Load the network once
@st.cache_resource
def load_net():
    # Ensure graph_opt.pb is in your main directory
    return cv2.dnn.readNetFromTensorflow("graph_opt.pb")

net = load_net()

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.thres = 0.2 # Default threshold

    def recv(self, frame):
        # Convert the frame to a NumPy array (BGR format for OpenCV)
        img = frame.to_ndarray(format="bgr24")
        frameWidth, frameHeight = img.shape[1], img.shape[0]

        # AI Inference
        net.setInput(cv2.dnn.blobFromImage(img, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]
        
        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > self.thres else None)
            
        # Draw the Skeleton
        for pair in POSE_PAIRS:
            partFrom, partTo = pair[0], pair[1]
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(img, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(img, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        # Return the processed frame back to the live stream
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("👤 Live Human Pose Estimator")
st.write("Real-time skeletal mapping via WebRTC.")

# Launch the streamer
ctx = webrtc_streamer(
    key="pose-estimation",
    video_processor_factory=PoseProcessor,
    rtc_configuration={ # This config helps bypass common firewall issues
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

# Allow users to adjust threshold live
if ctx.video_processor:
    ctx.video_processor.thres = st.slider("Detection Sensitivity", 0.0, 1.0, 0.2, 0.05)
