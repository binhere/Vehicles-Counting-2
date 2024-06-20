import cv2
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO
import torch
import os
from collections import defaultdict


def hex_to_rgba(hex_color, alpha=0.5):   
    hex_color = hex_color.lstrip('#')
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    rgba = f'rgba({red}, {green}, {blue}, {alpha})'
    return rgba


def extract_color_pos(df, max_width, max_height, filename="config.txt"):   
    with open(filename, 'w') as f:  
        configuration = ""
        for i, row in df.iterrows():
            r, g, b, alpha = row['fill'].strip('rgba()').split(', ')
            raw_pos = row['path'].strip('[]').split('], [')
            handled_pos = ""
            for string in raw_pos:
                parts = string.split(', ')
                if parts[0] != "'z'":
                    handled_pos += f'{float(parts[1]) / max_width},{float(parts[2]) / max_height};' 
            configuration += f'{r},{g},{b},{alpha}' + ";" + handled_pos + "\n"
        f.write(configuration)


def compute_center_of_zone(pts):
    x_sum = sum(pt[0] for pt in pts)
    y_sum = sum(pt[1] for pt in pts)
    x_avg = int(x_sum / len(pts))
    y_avg = int(y_sum / len(pts))
    return (x_avg, y_avg)


def draw_zones(frame, id_color_zone_pairs, counter, border_thickness=2):
    alpha = 1
    overlay = frame.copy()
    for pair in id_color_zone_pairs:
        id = pair[0]
        r, g, b, alpha = pair[1]
        pts = pair[2]
        center_x, center_y = compute_center_of_zone(pts)
        pts = np.array(pts)
        cv2.polylines(frame, [pts], isClosed=True, color=(b, g, r), thickness=border_thickness)
        cv2.fillPoly(overlay, pts=[pts], color=(b, g, r))
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    for pair in id_color_zone_pairs:
        id = pair[0]
        r, g, b, alpha = pair[1]
        pts = pair[2]
        center_x, center_y = compute_center_of_zone(pts)
        cv2.putText(frame, f'Zone {id+1}: {counter[id]}', (center_x-10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def load_zones(width, height, filename='config.txt'):
    id_color_zone_pairs = []
    with open(filename, 'r') as f:
        all_zones = f.read().split('\n')[:-1]
        for id, zone in enumerate(all_zones):
            parts = zone.split(';')
            color, all_points = parts[0], parts[1:-1]
            r, g, b, alpha = map(float, color.split(','))
            pts = []
            for point in all_points:
                x, y = map(float, point.split(','))
                x = int(x * width)
                y = int(y * height)
                pts.append([x, y])
            id_color_zone_pairs.append([id, (int(r), int(g), int(b), alpha), pts])
    return id_color_zone_pairs


def calculate_distance(position1, position2):
    return ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2) ** 0.5


def update_heatmap(heatmap, x, y, w, h, id_box, track_history):
    current_position = (x, y)
    top_left_x = max(0, int(x - w / 2))
    top_left_y = max(0, int(y - h / 2))
    bottom_right_x = min(heatmap.shape[1], int(x + w / 2))
    bottom_right_y = min(heatmap.shape[0], int(y + h / 2))
    track_history[id_box].append(current_position)
    if len(track_history[id_box]) >= 2:
        last_position = track_history[id_box][-2]
        track_history[id_box].pop(0)
        if calculate_distance(last_position, current_position) >= 5:
            heatmap[top_left_y:bottom_right_y, top_left_x:bottom_right_x] += 1
    return heatmap, track_history


def save_heatmap(frame, heatmap, filename='heatmap.png', alpha=0.7):
    heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite(filename, overlay)


def save_result(counter):
    with open('result/result.txt', 'w') as f:
        for key, value in counter.items():
            f.write(f'Zone {key+1}: {value}\n')