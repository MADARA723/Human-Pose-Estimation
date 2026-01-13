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

# Load the network once using @st.cache_resource to optimize memory
@st.cache_resource
def load_net():
    return cv2.dnn.readNetFromTensorflow("graph_opt.pb")

net = load_net()

st.title("Human Pose Estimation")
st.text('Upload a clear image to see the AI map skeletal keypoints.')

# --- IMAGE UPLOADER ---
img_file_buffer = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", 'png'])

# Use a demo image if nothing is uploaded
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    # Ensure you have 'stand.jpg' in your repo or change this to a file you have
    demo_image = 'stand.jpg' 
    try:
        image = np.array(Image.open(demo_image))
    except FileNotFoundError:
        st.warning("Please upload an image to begin.")
        st.stop()

st.subheader('Original Image')
st.image(image, caption="Original Image", use_container_width=True) 

# Threshold slider for sensitivity
thres = st.slider('Detection Threshold', 0, 100, 20) / 100

@st.cache_data
def poseDetector(frame, threshold):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    
    # AI Inference logic
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)
        
    # Draw the skeleton on the frame
    for pair in POSE_PAIRS:
        partFrom, partTo = pair[0], pair[1]
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.circle(frame, points[idFrom], 3, (0, 0, 255), thickness=-1)
            cv2.circle(frame, points[idTo], 3, (0, 0, 255), thickness=-1)
            
    return frame

# Run detection
with st.spinner("Analyzing pose..."):
    output = poseDetector(image.copy(), thres)

st.subheader('Positions Estimated')
st.image(output, caption="Skeletal Mapping Result", use_container_width=True)
