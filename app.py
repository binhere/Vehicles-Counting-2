import cv2
import streamlit as st
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from utils import *
import os
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import tempfile

#################### SET UP ####################

MAX_HEIGHT = 400
MAX_WIDTH = 600

global TRACKER_BYTETRACK, PATH_TRACKER, TRACKER, PATH_MODEL, MODEL_NAME, PATH_VID

PATH_TRACKER = 'tracker'
PATH_BYTETRACK = os.path.join(PATH_TRACKER, 'bytetrack.yaml')

MODEL_NAME = 'hcmdata_8n_50e.pt'   # YOLOv8 custom 
PATH_YOLO_WORLD_MODEL = os.path.join('model', 'yolov8s-worldv2.pt')  # Path to Yolo-World model
PATH_MODEL = os.path.join('model', MODEL_NAME)


# canvas
drawing_mode = 'polygon'
stroke_width = 3
if drawing_mode == "point":
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
color = st.sidebar.color_picker("Color", "#26eb50")
canvas_result = None

upload_video = st.file_uploader("Chọn tệp video", type=["mp4", "avi", "mov", "mkv"])

# config
selected_model = st.sidebar.selectbox("model", ("YOLOv8n-custom", "Yolo-World"))

selected_tracker = st.sidebar.selectbox("tracker", ("bytetrack", "botsort"))

conf_score = st.sidebar.slider(label='confidence', min_value=0.0, max_value=1.0)

clicked_button = st.sidebar.button('stop video')

# Input for custom classes for Yolo-World
custom_classes_input = None
if selected_model == 'Yolo-World':
    custom_classes_input = st.text_input("Nhập các lớp (phân cách bằng dấu phẩy, ví dụ: person,bus)")

# Create a canvas component
if upload_video is not None and (selected_model != 'Yolo-World' or custom_classes_input):
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload_video.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    progress_text = "Operation in progress. Please wait."
    succ, frame = cap.read()
    canvas_result = st_canvas(
        fill_color=hex_to_rgba(color),
        stroke_width=stroke_width,
        stroke_color=color,
        update_streamlit=True,
        background_image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        height=MAX_HEIGHT,
        width=MAX_WIDTH,
        drawing_mode="polygon",
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        display_toolbar=True,
        key="full_app",
    )
    
    if canvas_result is not None and canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")
        
        if len(objects.columns.to_list()):
            objects = objects.loc[:, ['fill', 'path']]
            extract_color_pos(objects, MAX_WIDTH, MAX_HEIGHT)

    if st.button("process"):
        frame_placeholder = st.empty()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if (selected_model == 'YOLOv8n-custom' or selected_model == 'Yolo-World') and selected_tracker in ['bytetrack']:
            
            if selected_model == 'YOLOv8n-custom':
                MODEL_NAME = 'hcmdata_8n_50e.pt'
                PATH_MODEL = os.path.join('model', MODEL_NAME)
                model = YOLO(PATH_MODEL).to(device)
            elif selected_model == 'Yolo-World':
                PATH_MODEL = PATH_YOLO_WORLD_MODEL
                model = YOLO(PATH_MODEL).to(device)
                if custom_classes_input:
                    custom_classes = [cls.strip() for cls in custom_classes_input.split(',')]
                    model.set_classes(custom_classes)
            
            if selected_tracker == 'bytetrack':
                TRACKER = PATH_BYTETRACK

            id_color_zone_pairs = load_zones(frame_width, frame_height)

            id_counted = []
            counter = {i: 0 for i in range(len(id_color_zone_pairs))}
            
            fps_interval = 5  # Display FPS every 5 frames
            frame_count = 0
            total_fps = 0
        
            while(cap.isOpened()):
                ret, frame = cap.read()
                
                if ret:  
                    results = model.track(frame, persist=True, verbose=False, tracker=TRACKER, conf=conf_score)
                    annotated_frame = results[0].plot()
                    
                    if results[0].boxes.is_track == False:
                        continue
                    
                    for box in results[0].boxes.cpu().numpy():
                        x, y, w, h = box.xywh.astype(int)[0]
                        id_box = box.id.astype(int)[0]
                        id_cls = box.cls.astype(int)[0]
                        
                        for pair in id_color_zone_pairs:
                            id_zone = pair[0]
                            zone = pair[2]
                            
                            pts = np.array(zone, np.int32)
                            center_point = (float(x), float(y))
                            dist = cv2.pointPolygonTest(pts, center_point, False)

                            if dist > 0:
                                if id_box not in id_counted:
                                    id_counted.append(id_box)
                                    counter[id_zone] += 1                        
                                    break
                                
                            if id_box not in id_counted:
                                annotated_frame = cv2.circle(annotated_frame, (x, y), radius=2, color=(0, 255, 255), thickness=6)
                            else:
                                annotated_frame = cv2.circle(annotated_frame, (x, y), radius=2, color=(0, 255, 0), thickness=6)
                                
                    annotated_frame = draw_zones(annotated_frame, id_color_zone_pairs, counter)
                    
                    frame_count += 1
                    fps = 1 / (sum(results[0].speed.values())/1000)
                    total_fps += fps
                    if frame_count % fps_interval == 0:
                        avg_fps = total_fps / fps_interval
                        fps_text = f'FPS: {avg_fps:.2f}'
                        total_fps = 0
                        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    total_vehicles = [val for key, val in counter.items()]
                    cv2.putText(annotated_frame, f'Total: {sum(total_vehicles)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    frame_placeholder.image(annotated_frame, channels="BGR")
                    
                    if cv2.waitKey(1) == ord('q') or clicked_button:
                        save_result(counter)
                        print('>> saved result')
                        print('>> exiting')
                        break
                    
                else:                    
                    save_result(counter)
                    print('>> saved result')
                    print('>> video ended')
                    break
                
            cap.release()
            cv2.destroyAllWindows()
            os.remove(video_path)
