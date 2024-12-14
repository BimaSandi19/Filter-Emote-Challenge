import cv2
import mediapipe as mp
import random
import time
import os

class EmoteChallenge:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.cap = cv2.VideoCapture(0)
        self.score = 0
        self.game_duration = 60  # 60 detik
        self.start_time = None
        
        # Menentukan emoji dasar dan pola landmark wajah yang sesuai
        self.emojis = {
            "emoji1": "emoji1.png",
            "emoji2": "emoji2.png",
            "emoji3": "emoji3.png",
            "emoji4": "emoji4.png",
            "emoji5": "emoji5.png"
        }
        self.current_emoji = None
        self.emoji_images = self.load_emoji_images()

    def load_emoji_images(self):
        emoji_images = {}
        for key, filename in self.emojis.items():
            path = os.path.join('assets', filename)
            emoji_images[key] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return emoji_images

    def detect_expression(self, face_landmarks):
        if not face_landmarks or len(face_landmarks.landmark) < 468:
            return None  # Tidak ada data valid
        
        # Landmark penting untuk deteksi ekspresi
        left_lip_corner = face_landmarks.landmark[61]
        right_lip_corner = face_landmarks.landmark[291]
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        left_eye_top = face_landmarks.landmark[159]
        left_eye_bottom = face_landmarks.landmark[145]
        right_eye_top = face_landmarks.landmark[386]
        right_eye_bottom = face_landmarks.landmark[374]
        nose_tip = face_landmarks.landmark[1]
        left_eyebrow = face_landmarks.landmark[70]
        right_eyebrow = face_landmarks.landmark[300]
        tongue_tip = face_landmarks.landmark[19]  # Landmark lidah (asumsi)

        # 1. Deteksi Senyum
        lip_width = ((right_lip_corner.x - left_lip_corner.x) ** 2 + (right_lip_corner.y - left_lip_corner.y) ** 2) ** 0.5
        lip_height = ((upper_lip.x - lower_lip.x) ** 2 + (upper_lip.y - lower_lip.y) ** 2) ** 0.5
        if lip_width / lip_height > 2.0:
            return "emoji_senyum"  # Emoji senyum

        # 2. Deteksi Mulut Terbuka
        if lip_height / lip_width > 0.6:
            return "emoji_mulut_terbuka"  # Emoji mulut terbuka

        # 3. Deteksi Mata Tertutup (Mengedip)
        left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
        if left_eye_height < 0.01 and right_eye_height < 0.01:
            return "emoji_mengedip"  # Emoji mata tertutup

        # 4. Deteksi Lidah Menjulur
        tongue_height = abs(tongue_tip.y - lower_lip.y)
        lip_to_tongue_threshold = 0.05  # Toleransi jarak untuk lidah menjulur
        if tongue_height > lip_to_tongue_threshold:
            return "emoji_lidah_menjulur"  # Emoji lidah menjulur

        # 5. Deteksi Mengangkat Alis
        eyebrow_to_eye_distance = abs(left_eyebrow.y - left_eye_top.y)
        eyebrow_threshold = 0.02  # Toleransi untuk jarak alis dan mata
        if eyebrow_to_eye_distance > eyebrow_threshold:
            return "emoji_angkat_alis"  # Emoji mengangkat alis

        # 6. Deteksi Mengerutkan Dahi
        eyebrow_distance = abs(left_eyebrow.x - right_eyebrow.x)
        if eyebrow_distance < 0.05:
            return "emoji_mengerutkan_dahi"  # Emoji mengerutkan dahi

        # 7. Deteksi Hidung Berkerut
        nose_distance = abs(upper_lip.y - nose_tip.y)
        nose_wrinkle_threshold = 0.02  # Toleransi untuk jarak hidung dan bibir
        if nose_distance < nose_wrinkle_threshold:
            return "emoji_hidung_kerut"  # Emoji hidung berkerut

        # 8. Deteksi Mata Terbuka Lebar
        if left_eye_height > 0.03 and right_eye_height > 0.03:
            return "emoji_mata_terbuka_lebar"  # Emoji mata terbuka lebar

        # 9. Deteksi Menoleh ke Kanan
        if right_lip_corner.x - nose_tip.x > 0.1:
            return "emoji_menoleh_kanan"  # Emoji menoleh ke kanan

        # 10. Deteksi Menoleh ke Kiri
        if nose_tip.x - left_lip_corner.x > 0.1: # type: ignore
            return "emoji_menoleh_kiri"  # Emoji menoleh ke kiri

        # Tidak ada ekspresi terdeteksi
        return None

    def overlay_emoji(self, frame, emoji_image, position):
        x, y = position
        h, w, _ = emoji_image.shape
        alpha_emoji = emoji_image[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_emoji

        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = (alpha_emoji * emoji_image[:, :, c] +
                                      alpha_frame * frame[y:y+h, x:x+w, c])
        return frame

    def draw_text_with_background(self, frame, texts, position, font, scale, text_color, bg_color, thickness):
        max_text_width = 0
        total_text_height = 0
        text_sizes = []

        for text in texts:
            (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
            max_text_width = max(max_text_width, text_width)
            total_text_height += text_height + baseline
            text_sizes.append((text_width, text_height, baseline))

        x, y = position
        cv2.rectangle(frame, (x, y), (x + max_text_width, y + total_text_height), bg_color, cv2.FILLED)

        y_offset = y
        for i, text in enumerate(texts):
            text_width, text_height, baseline = text_sizes[i]
            cv2.putText(frame, text, (x, y_offset + text_height), font, scale, text_color, thickness)
            y_offset += text_height + baseline

    def run_game(self):
        self.start_time = time.time()
        self.score = 0  # Pastikan skor dimulai dari 0
        
        while True:
            # Menghitung sisa waktu
            elapsed_time = int(time.time() - self.start_time)
            remaining_time = max(0, self.game_duration - elapsed_time)
            
            if remaining_time == 0:
                break

            # Membaca frame dari kamera
            ret, frame = self.cap.read()
            if not ret:
                break

            # Membalik frame secara horizontal
            frame = cv2.flip(frame, 1)

            # Mengubah ke RGB untuk MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            # Jika tidak ada emoji saat ini, pilih secara acak
            if not self.current_emoji:
                self.current_emoji = random.choice(list(self.emojis.keys()))
                self.emoji_start_time = time.time()
            # Menampilkan emoji saat ini
            emoji_image = self.emoji_images[self.current_emoji]
            frame = self.overlay_emoji(frame, emoji_image, (frame.shape[1] - emoji_image.shape[1] - 10, 10))

            # Menampilkan skor dan waktu
            self.draw_text_with_background(frame, [f"Score: {self.score}", f"Time: {remaining_time}s"], (50, 50),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), (255, 255, 255), 2)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    detected_expression = self.detect_expression(face_landmarks)
                    if detected_expression == self.current_emoji:
                        self.score += 100
                        self.current_emoji = None  # Pilih emoji baru
                        self.emoji_start_time = time.time()

            cv2.imshow('Emote Challenge', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Permainan Selesai! Skor Akhir: {self.score}")

if __name__ == "__main__":
    game = EmoteChallenge()
    game.run_game()
