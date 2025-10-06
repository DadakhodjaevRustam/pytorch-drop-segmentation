import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import segmentation_models_pytorch as smp

# Инициализация переменных для ROI
roi_selected = False
roi_box = None
drawing = False
paused = False

# Функция обработки событий мыши для выбора ROI
def select_roi(event, x, y, flags, param):
    global roi_box, drawing, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_box = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_box[2] = x
            roi_box[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_selected = True
        roi_box[2] = x
        roi_box[3] = y
        if roi_box[0] > roi_box[2]:
            roi_box[0], roi_box[2] = roi_box[2], roi_box[0]
        if roi_box[1] > roi_box[3]:
            roi_box[1], roi_box[3] = roi_box[3], roi_box[1]

# Устройство для вычислений
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры модели
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d'
CLASSES = ["background", "drop"]

# Инициализация модели
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
model_path = r'...' # Путь к папке с моделью
state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)

# Функции обработки кадра
def preprocess_frame(frame, target_size=(512 * 2, 512 * 2)):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_size = (frame.shape[1], frame.shape[0])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image_rgb).unsqueeze(0)
    return image.to(DEVICE), original_size

def postprocess_output(output, original_size):
    output = output.squeeze().cpu().detach().numpy()
    mask = np.argmax(output, axis=0)
    mask = cv2.resize(mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    return mask

def create_colored_mask(mask, colors=None):
    if colors is None:
        colors = {0: (0, 0, 0), 1: (0, 0, 255)}
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in colors.items():
        colored_mask[mask == class_id] = color
    return colored_mask

def apply_roi(frame, mask, roi):
    colored_mask = create_colored_mask(mask)
    if roi is None:
        result = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
        return result
    x1, y1, x2, y2 = roi
    roi_frame = frame.copy()
    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    colored_roi_mask = create_colored_mask(roi_mask)
    result = cv2.addWeighted(frame, 0.7, colored_roi_mask, 0.3, 0)
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return result

# Функция обновления кадра
def update_frame(cap, pos=None):
    if pos is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if not ret:
        return None, None
    if roi_selected:
        x1, y1, x2, y2 = roi_box
        roi_frame = frame[y1:y2, x1:x2].copy()
        if roi_frame.size > 0:
            image_tensor, original_size = preprocess_frame(roi_frame)
            with torch.no_grad():
                output = model(image_tensor)
            mask = postprocess_output(output, (x2 - x1, y2 - y1))
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask
            result = apply_roi(frame, full_mask, roi_box)
        else:
            result = frame.copy()
    else:
        image_tensor, original_size = preprocess_frame(frame)
        with torch.no_grad():
            output = model(image_tensor)
        mask = postprocess_output(output, original_size)
        result = apply_roi(frame, mask, None)
    return result, int(cap.get(cv2.CAP_PROP_POS_FRAMES))

# Функция обработки трекбара
def on_trackbar(pos):
    global paused
    paused = True
    result, current_frame = update_frame(cap, pos)
    if result is not None:
        display_frame = result.copy()
        cv2.putText(display_frame, f"Frame: {current_frame}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Droplet Detection', display_frame)

# Загрузка видео
video_path = r'...' # Путь к папке с видео
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Не удалось открыть видео файл")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Видео: {width}x{height}, {fps:.2f} FPS, Всего кадров: {total_frames}")

# Получаем первый кадр для выбора ROI
ret, frame = cap.read()
if not ret:
    raise ValueError("Не удалось прочитать первый кадр")

# Создаем окно и устанавливаем обработчик мыши
cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Select ROI', select_roi)

# Выбор ROI
while True:
    display_frame = frame.copy()
    if roi_box is not None:
        cv2.rectangle(display_frame, (roi_box[0], roi_box[1]),
                      (roi_box[2], roi_box[3]), (0, 255, 0), 2)
    cv2.imshow('Select ROI', display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and roi_selected:
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow('Select ROI')

# Создаем основное окно с трекбаром
cv2.namedWindow('Droplet Detection', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Frame', 'Droplet Detection', 0, total_frames - 1, on_trackbar)

# Основной цикл обработки
last_display_frame = None  # Переменная для хранения последнего валидного кадра
while True:
    if not paused:
        result, current_frame = update_frame(cap)
        if result is None:
            break
        cv2.setTrackbarPos('Frame', 'Droplet Detection', current_frame)
        display_frame = result.copy()
        cv2.putText(display_frame, f"Frame: {current_frame}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Space: Pause/Resume | Q: Quit | F/B: Forward/Backward | S: Save", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Droplet Detection', display_frame)
        last_display_frame = display_frame  # Сохраняем текущий кадр

    key = cv2.waitKey(25) & 0xFF
    if key == ord(' '):
        paused = not paused
    elif key == ord('q'):
        break
    elif key == ord('f'):
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        new_pos = min(current_pos + 10, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        paused = True
        on_trackbar(new_pos)
    elif key == ord('b'):
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        new_pos = max(current_pos - 10, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        paused = True
        on_trackbar(new_pos)
    elif key == ord('s'):
        if last_display_frame is not None:
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            save_path = f"frame_{current_pos}.png"
            cv2.imwrite(save_path, last_display_frame)
            print(f"Сохранен кадр {current_pos} в {save_path}")
        else:
            print("Ошибка: нет кадра для сохранения")

cap.release()
cv2.destroyAllWindows()