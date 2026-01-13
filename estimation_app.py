import streamlit as st
from PIL import Image
import numpy as np
import cv2

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

# Loading the network - using @st.cache_resource to stop "Dosing Out" the CPU
@st.cache_resource
def load_net():
    return cv2.dnn.readNetFromTensorflow("graph_opt.pb")

net = load_net()

st.title("Human Pose Estimation")
st.markdown("### Interactive Skeletal Mapping")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Settings")
mode = st.sidebar.radio("Choose Input:", ("Upload Image", "Live Webcam Snapshot"))
thres = st.sidebar.slider('Detection Threshold', 0, 100, 20) / 100

def poseDetector(frame):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
        
    for pair in POSE_PAIRS:
        partFrom, partTo = pair[0], pair[1]
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    return frame

# --- MAIN LOGIC ---
source = None
if mode == "Upload Image":
    source = st.file_uploader("Upload a clear image", type=["jpg", "jpeg", 'png'])
else:
    source = st.camera_input("Take a photo to estimate pose")

if source is not None:
    # Read the image
    image = np.array(Image.open(source))
    
    # Process
    st.subheader('Estimated Position')
    with st.spinner("Analyzing geometry..."):
        # use_container_width=True fixes the deprecation warning in your logs
        output = poseDetector(image.copy())
        st.image(output, use_container_width=True) 
else:
    st.info("Please provide an image or take a snapshot to see the AI in action.")
