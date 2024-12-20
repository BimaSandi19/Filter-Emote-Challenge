import cv2
import os

def load_emoji_images(emojis):
    """
    Memuat gambar emoji dari direktori assets.
    
    Parameters:
    emojis (dict): Kamus emoji dan nama file gambar.
    
    Returns:
    dict: Kamus emoji dan gambar yang dimuat.
    """
    emoji_images = {}
    for key, filename in emojis.items():
        path = os.path.join('assets', filename)
        emoji_images[key] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return emoji_images

def overlay_emoji(frame, emoji_image, position):
    """
    Menambahkan gambar emoji pada frame.
    
    Parameters:
    frame (ndarray): Frame gambar.
    emoji_image (ndarray): Gambar emoji.
    position (tuple): Posisi emoji pada frame.
    
    Returns:
    ndarray: Frame dengan emoji yang ditambahkan.
    """
    x, y = position
    h, w, _ = emoji_image.shape
    alpha_emoji = emoji_image[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_emoji

    for c in range(0, 3):
        frame[y:y+h, x:x+w, c] = (alpha_emoji * emoji_image[:, :, c] +
                                  alpha_frame * frame[y:y+h, x:x+w, c])
    return frame