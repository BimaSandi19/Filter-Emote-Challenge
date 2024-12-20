import cv2

def distance(point1, point2):
    """
    Menghitung jarak antara dua titik.
    
    Parameters:
    point1 (object): Titik pertama.
    point2 (object): Titik kedua.
    
    Returns:
    float: Jarak antara dua titik.
    """
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

def calculate_gradient(p1, p2):
    """
    Menghitung gradien antara dua titik.
    
    Parameters:
    p1 (object): Titik pertama.
    p2 (object): Titik kedua.
    
    Returns:
    float: Gradien antara dua titik.
    """
    return (p2.y - p1.y) / (p2.x - p1.x) if p2.x != p1.x else 0

def draw_text_with_background(frame, texts, position, font, scale, text_color, bg_color, thickness):
    """
    Menggambar teks dengan latar belakang pada frame.
    
    Parameters:
    frame (ndarray): Frame gambar.
    texts (list): Daftar teks yang akan digambar.
    position (tuple): Posisi teks.
    font (int): Jenis font.
    scale (float): Skala teks.
    text_color (tuple): Warna teks.
    bg_color (tuple): Warna latar belakang.
    thickness (int): Ketebalan teks.
    """
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