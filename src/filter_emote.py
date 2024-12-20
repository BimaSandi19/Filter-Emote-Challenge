import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import mediapipe as mp
import random
import time
import threading
from utils import distance, calculate_gradient, draw_text_with_background, display_final_score
from emoji import load_emoji_images, overlay_emoji

class EmoteChallenge:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.cap = cv2.VideoCapture(0)
        self.score = 0
        self.game_duration = 30  # 30 detik
        self.start_time = None
        self.successful_emojis = []  # Menyimpan emoji yang berhasil ditiru
        self.used_emojis = set()  # Menyimpan emoji yang sudah muncul
        
        # Menentukan emoji dasar dan pola landmark wajah yang sesuai
        self.emojis = {
            "emoji1": "emoji1.png",
            "emoji2": "emoji2.png",
            "emoji3": "emoji3.png",
            "emoji4": "emoji4.png",
            "emoji5": "emoji5.png"
        }
        self.current_emoji = None
        self.emoji_images = load_emoji_images(self.emojis)
        self.frame = None
        self.running = True

    def detect_expression(self, face_landmarks):
        # Mengambil landmark yang diperlukan
        landmarks = face_landmarks.landmark

        # Landmark untuk bibir, mata, dan alis
        left_lip_corner = landmarks[61]
        right_lip_corner = landmarks[291]
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]

        # Menghitung jarak yang diperlukan
        lip_width = distance(left_lip_corner, right_lip_corner)
        lip_height = distance(upper_lip, lower_lip)
        left_eye_height = distance(left_eye_top, left_eye_bottom)
        right_eye_height = distance(right_eye_top, right_eye_bottom)

        # Gradien bibir untuk mendeteksi lengkungan
        left_gradient = calculate_gradient(left_lip_corner, upper_lip)
        right_gradient = calculate_gradient(right_lip_corner, upper_lip)

        # Debug print untuk membantu memahami nilai yang dihitung
        print(f"lip_width: {lip_width}, lip_height: {lip_height}, left_eye_height: {left_eye_height}, right_eye_height: {right_eye_height}")
        print(f"left_gradient: {left_gradient}, right_gradient: {right_gradient}")

        # Thresholds untuk mendeteksi ekspresi
        thresholds = {
            "mouth_open": 0.03,
            "eye_closed": 0.02,
            "eyebrow_raise": 0.03,
            "smile_width": 0.12,  # Menurunkan threshold untuk lebar senyum
            "smile_height": 0.025,  # Menurunkan threshold untuk tinggi senyum
            "lip_corner_down": -0.08,  # Menurunkan threshold untuk lengkungan bibir ke bawah
            "lip_corner_up": 0.2  # Menambahkan threshold untuk lengkungan bibir ke atas
        }

        # Deteksi ekspresi berdasarkan threshold
        if lip_width > thresholds["smile_width"] and lip_height > thresholds["smile_height"]:
            print("Detected expression: emoji5 (senyum lebar)")
            return "emoji5"  # Emoji senyum lebar hingga terlihat gigi
        elif (left_eye_height < thresholds["eye_closed"] or right_eye_height < thresholds["eye_closed"]) and lip_height > thresholds["mouth_open"]:
            print("Detected expression: emoji2 (mata mengedip 1 dan lidah melet)")
            return "emoji2"  # Emoji mata mengedip 1 dan lidah melet
        elif lip_height > thresholds["mouth_open"] and lip_width < thresholds["smile_width"]:
            print("Detected expression: emoji1 (bengong)")
            return "emoji1"  # Emoji bengong hingga mulut terbuka
        elif (left_gradient < thresholds["lip_corner_down"] or right_gradient < thresholds["lip_corner_down"]):
            print("Detected expression: emoji4 (sedih/murung)")
            return "emoji4"  # Emoji sedih/murung
        elif (lip_height < thresholds["mouth_open"] and lip_width < thresholds["smile_width"] and
            left_gradient > thresholds["lip_corner_down"] and left_gradient < thresholds["lip_corner_up"] and
            right_gradient > thresholds["lip_corner_down"] and right_gradient < thresholds["lip_corner_up"]):
            print("Detected expression: emoji3 (ekspresi netral)")
            return "emoji3"  # Emoji ekspresi netral

        print("No expression detected")
        return None  # Tidak ada ekspresi yang terdeteksi

    def capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = cv2.flip(frame, 1)

    def run_game(self):
        self.start_time = time.time()
        self.score = 0
        self.game_duration = 30
        self.used_emojis = set()
        self.successful_emojis = []

        # Pilih emoji pertama sebelum hitung mundur
        self.current_emoji = random.choice(list(self.emojis.keys()))
        self.used_emojis.add(self.current_emoji)

        # Mulai thread untuk menangkap frame
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.start()

        # Waktu persiapan 3 detik
        preparation_time = 3
        while preparation_time > 0:
            frame = cv2.imread('assets/background.jpg')
            emoji_image = self.emoji_images[self.current_emoji]
            frame = overlay_emoji(frame, emoji_image, (frame.shape[1] - emoji_image.shape[1] - 10, 10))
            draw_text_with_background(frame, [f"Prepare! Starting in {preparation_time}..."], (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), (255, 255, 255), 2)
            cv2.imshow('Emote Challenge', frame)
            cv2.waitKey(1000)
            preparation_time -= 1

        while True:
            elapsed_time = int(time.time() - self.start_time)
            remaining_time = max(0, self.game_duration - elapsed_time)

            if remaining_time == 0:
                break

            if self.frame is not None:
                frame = self.frame.copy()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if self.current_emoji is not None:
                    emoji_image = self.emoji_images[self.current_emoji]
                    frame = overlay_emoji(frame, emoji_image, (frame.shape[1] - emoji_image.shape[1] - 10, 10))

                draw_text_with_background(frame, [f"Score: {self.score}", f"Time: {remaining_time}s"], (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), (255, 255, 255), 2)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        detected_expression = self.detect_expression(face_landmarks)
                        if detected_expression == self.current_emoji:
                            self.score += 100
                            self.successful_emojis.append(self.current_emoji)

                            remaining_emojis = list(set(self.emojis.keys()) - self.used_emojis)
                            if remaining_emojis:
                                self.current_emoji = random.choice(remaining_emojis)
                                self.used_emojis.add(self.current_emoji)
                            else:
                                self.current_emoji = None

                if self.current_emoji is None:
                    break

                cv2.imshow('Emote Challenge', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.running = False
        capture_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Permainan Selesai! Skor Akhir: {self.score}")
        print("Emoji yang berhasil ditiru:", self.successful_emojis)
    
        # Setelah game selesai, tampilkan skor akhir
        final_frame = cv2.imread('assets/bg2.jpg')
        display_final_score(final_frame, self.score)

if __name__ == "__main__":
    game = EmoteChallenge()
    game.run_game()