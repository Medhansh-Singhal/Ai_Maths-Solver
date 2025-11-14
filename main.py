import cv2
from datetime import datetime as dt
import time
import mediapipe as mp
import numpy as np
from PIL import Image
import streamlit as st
import google.generativeai as genai
import os
import dotenv

dotenv.load_dotenv()

genai.configure(api_key="YOUR-GEMINI-API-KEY")
model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(layout="wide")
st.title("AI Maths Solver")
col1, col2 = st.columns([2, 1])
with col1:
    run = st.button("Run")
    if run:
        frame_window = st.image([])
    else:
        frame_window = st.empty()
with col2:
    st.title("Results")
    output_text = st.subheader("")

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
tipids = [4, 8, 12, 16, 20]
eraserthickness = 30
brushthickness = 6
colour = (255, 255, 0)
prev_pos = None
img_canvas = np.zeros((720, 1280, 3), np.uint8)
result_text = ""
fontscale = 2
thickness = 5
color_bg = (135, 249, 8)  # for text
color_out = (0, 0, 0)  # for text
textcolor = (255, 0, 0)  # for text
folder = "Storage"
database = f"{folder}/Image{time.time()}.jpg"
count = 1
image_list = []
past_image_list = []

def fingerinfo(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img_copy, hand_landmark, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=-1, circle_radius=6),
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3))
            landmarks = []
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, c = img_rgb.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])
                # print(landmarks)
            open_finger = []
            if landmarks[tipids[0]][1] < landmarks[tipids[0] - 1][1]:
                open_finger.append(1)
            else:
                open_finger.append(0)

            for i in range(1, 5):
                if landmarks[tipids[i]][2] < landmarks[tipids[i] - 2][2]:
                    open_finger.append(1)
                else:
                    open_finger.append(0)
            # print(open_finger)
        return open_finger, landmarks
    else:
        return None


def drawing_mode(finger_info, prev_pos, img_canvas, colour, count, image_list):
    finger_up, landmarks = finger_info
    current_pos = None
    if finger_up == [0, 1, 0, 0, 0] or finger_up == [1, 1, 0, 0, 0]:
        count = 1
        # print(count)
        current_pos = landmarks[8][1:]
        if prev_pos == None: prev_pos = current_pos
        if colour == (0, 0, 0):
            cv2.line(img_copy, current_pos, prev_pos, colour, eraserthickness)
            cv2.line(img_canvas, current_pos, prev_pos, colour, eraserthickness)
        else:
            cv2.line(img_copy, current_pos, prev_pos, colour, brushthickness)
            cv2.line(img_canvas, current_pos, prev_pos, colour, brushthickness)
    else:
        # print(count)
        count = count + 1
        if count == 2:
            os.makedirs(folder, exist_ok=True)
            timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{folder}/Image_{timestamp}.jpg"
            success = cv2.imwrite(filename, img_canvas)
            if success:
                print(f"Saved Image - {filename}")
            else:
                print("Failed to save Image.")
    names = os.listdir("Storage")
    image_list = [name for name in names if name.startswith("Image_")]

    return current_pos, img_canvas, count, image_list

def ai_analyze(model, img_canvas, finger_info):
    finger_up, landmarks = finger_info
    if landmarks[8][2] < 100 and landmarks[12][2] < 100:
        if 980 < landmarks[8][1] < 1280 and 980 < landmarks[12][1] < 1280:
            (text_w, text_h), baseline = cv2.getTextSize("RESULT", cv2.FONT_HERSHEY_SIMPLEX
                                                         , fontscale, thickness)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_bg, cv2.FILLED)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_out, 2)
            cv2.putText(img_copy, "RESULT", ((1120 - text_w // 2), 702),
                        cv2.FONT_HERSHEY_SIMPLEX, fontscale, textcolor,
                        thickness, cv2.LINE_AA)
            pil_img = Image.fromarray(img_canvas)
            response = model.generate_content(
                ["Analyze this image and generate output about it to the point"
                 " and if it a math or physics probelm then solve it",
                 pil_img])
            print(response.text)
            os.makedirs(folder, exist_ok=True)
            timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{folder}/Results/Image_{timestamp}.jpg"
            cv2.imwrite(filename, img_canvas)
            # f = open(f"{folder}/Results/Image_{timestamp}.txt", "w")
            # f.write(response.text)
            # f.close()
            return response.text


def selecting_mode(finger_info, colour, img_canvas):
    finger_up, landmarks = finger_info
    # current_pos = None
    if landmarks[8][2] < 100 and landmarks[12][2] < 100:
        if landmarks[8][1] < 185 and landmarks[12][1] < 185:
            (text_w, text_h), baseline = cv2.getTextSize("CLEAR", cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_bg, cv2.FILLED)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_out, 2)
            cv2.putText(img_copy, "CLEAR", ((1120 - text_w // 2), 702), cv2.FONT_HERSHEY_SIMPLEX, fontscale, textcolor,
                        thickness,
                        cv2.LINE_AA)
            img_canvas = np.zeros_like(img_copy)
            colour = (255, 255, 0)

        if 185 < landmarks[8][1] < 370 and 185 < landmarks[12][1] < 370:
            (text_w, text_h), baseline = cv2.getTextSize("BLUE", cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_bg, cv2.FILLED)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_out, 2)
            cv2.putText(img_copy, "BLUE", ((1120 - text_w // 2), 702), cv2.FONT_HERSHEY_SIMPLEX, fontscale, textcolor,
                        thickness,
                        cv2.LINE_AA)
            colour = (255, 255, 0)
        if 370 < landmarks[8][1] < 560 and 370 < landmarks[12][1] < 560:
            (text_w, text_h), baseline = cv2.getTextSize("RED", cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_bg, cv2.FILLED)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_out, 2)
            cv2.putText(img_copy, "RED", ((1120 - text_w // 2), 702), cv2.FONT_HERSHEY_SIMPLEX, fontscale, textcolor,
                        thickness,
                        cv2.LINE_AA)
            colour = (0, 0, 255)
        if 560 < landmarks[8][1] < 755 and 560 < landmarks[12][1] < 755:
            (text_w, text_h), baseline = cv2.getTextSize("GREEN", cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_bg, cv2.FILLED)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_out, 2)
            cv2.putText(img_copy, "GREEN", ((1120 - text_w // 2), 702), cv2.FONT_HERSHEY_SIMPLEX, fontscale, textcolor,
                        thickness,
                        cv2.LINE_AA)
            colour = (0, 255, 0)
        if 755 < landmarks[8][1] < 980 and 755 < landmarks[12][1] < 980:
            (text_w, text_h), baseline = cv2.getTextSize("ERASER", cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_bg, cv2.FILLED)
            cv2.rectangle(img_copy, (970, 650), (1270, 710), color_out, 2)
            cv2.putText(img_copy, "ERASER", ((1120 - text_w // 2), 702), cv2.FONT_HERSHEY_SIMPLEX, fontscale, textcolor,
                        thickness,
                        cv2.LINE_AA)
            colour = (0, 0, 0)

    return colour, img_canvas


# def keyboard(finger_info, img_canvas, image_list):
#     finger_up, landmarks = finger_info
#     k = cv2.waitKey(1)
#     print(img_canvas.shape)
#     if finger_up != [0, 1, 0, 0, 0] and finger_up != [1, 1, 0, 0, 0] or finger_up is None:
#         if k == ord("u") or k == ord("U"):
#             undo_image = cv2.imread("AI Maths solver/Storage/0Default.jpg")
#             img_canvas = cv2.cvtColor(undo_image, cv2.COLOR_BGR2RGB)
#             print(img_canvas.shape)
#     return img_canvas

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))
    img_copy = img.copy()
    if not ret:
        print('Failed to grab frame')
        break
    if img_canvas is None:
        img_canvas = np.zeros_like(img_copy)

    finger_info = fingerinfo(img)
    if finger_info:
        finger_up, landmarks = finger_info
        # print(finger_up)
        # print(landmarks)
        colour, img_canvas = selecting_mode(finger_info, colour, img_canvas)
        prev_pos, img_canvas, count, image_list = drawing_mode(finger_info, prev_pos, img_canvas, colour, count,
                                                               image_list)
        # print(len(image_list))
        # img_canvas = keyboard(finger_info, img_canvas, image_list)
        result_text = ai_analyze(model, img_canvas, finger_info)
    control_button = cv2.imread("Control_Button.png")
    control_button_resized = cv2.resize(control_button, (1280, 100))
    img_copy = cv2.addWeighted(img_copy, 1, img_canvas, 1, 0)
    img_copy[0:100, 0:1280] = control_button_resized
    frame_window.image(img_copy, channels="BGR")
    if result_text:
        output_text.text(result_text)
    # cv2.imshow('OUTPUT WINDOW', img_copy)
    # cv2.imshow('CANVAS', img_canvas)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        if len(image_list) > 0:
            # j = 0
            past_image_list = ["Past_" + name for name in image_list]
            print(image_list)
            print(past_image_list)
            # path = "Ai Maths Solver/Storage"
            for current_name, new_name in zip(image_list, past_image_list):
                old_file_path = folder + '/' + current_name
                new_file_path = folder + '/' + new_name
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} to {new_file_path}")
        print('Escape hit, closing...')
        break

cam.release()
cv2.destroyAllWindows()

