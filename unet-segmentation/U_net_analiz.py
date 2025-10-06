import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp

# Устройство для вычислений
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры модели (должны совпадать с параметрами при обучении)
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

# Загрузка весов модели из .pth файла
model_path = r'...' # Путь к папке с моделью
state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()  # Переключение модели в режим оценки
model.to(DEVICE)


# Предобработка изображения
def preprocess_image(image_path, target_size=(512, 512)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение по пути: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертируем из BGR в RGB
    original_size = (image.shape[1], image.shape[0])  # Сохраняем исходный размер (width, height)

    # Преобразование с сохранением пропорций
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Изменяем размер
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация для ImageNet
    ])
    image_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension
    return image_tensor.to(DEVICE), original_size


# Постобработка результата
def postprocess_output(output, original_size):
    output = output.squeeze().cpu().detach().numpy()  # Убираем batch dimension и переносим на CPU
    mask = np.argmax(output, axis=0)  # Выбираем канал с наибольшей вероятностью
    mask = cv2.resize(mask.astype(np.uint8), original_size,
                      interpolation=cv2.INTER_NEAREST)  # Возвращаем к исходному размеру
    return mask


# Визуализация результата
def visualize_result(image, mask):
    # Включаем интерактивный режим для зума и прокрутки
    plt.ion()

    # Создаём фигуру с большим размером для детального просмотра
    plt.figure(figsize=(15, 5))

    # 1. Оригинальное изображение
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    # 2. Сегментационная маска
    plt.subplot(1, 3, 2)
    plt.title('Segmentation Mask')
    plt.imshow(mask, cmap='jet')  # Используем 'jet' для визуализации многоклассовой маски
    plt.axis('off')

    # 3. Наложение оригинального изображения и маски
    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(image)  # Оригинальное изображение
    plt.imshow(mask, cmap='jet', alpha=0.5)  # Накладываем маску с полупрозрачностью
    plt.axis('off')

    # Настраиваем отображение
    plt.tight_layout()
    plt.show()


# Путь к изображению
image_path = r'D:\Pycharm\Project_MAIN\segmentation_ai\segmentation_ai\Dataset\Test\11.jpg'

# Предобработка изображения
image_tensor, original_size = preprocess_image(image_path, target_size=(512 * 4, 512 * 4))

# Инференс модели
with torch.no_grad():
    output = model(image_tensor)

# Постобработка результата
mask = postprocess_output(output, original_size)

# Визуализация результата
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Конвертируем из BGR в RGB
visualize_result(original_image, mask)


# Анализ распределения капель по радиусам
def analyze_drop_sizes(mask):
    # Создаем бинарную маску для капель
    binary_mask = (mask == 1).astype(np.uint8) * 255

    # Находим контуры
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    radii = []
    for cnt in contours:
        # Пропускаем слишком маленькие контуры
        if cv2.contourArea(cnt) < 10:
            continue

        # Получаем минимальную охватывающую окружность
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        radii.append(radius)

    return radii


# Получаем радиусы капель
radii = analyze_drop_sizes(mask)

# Создаем фигуру с двумя колонками
plt.figure(figsize=(16, 8))

# Левая панель - оригинальное изображение с сеткой
ax1 = plt.subplot(1, 2, 1)
plt.imshow(original_image)

# Настраиваем сетку и оси
step = 100  # Шаг сетки в пикселях
x_max, y_max = original_image.shape[1], original_image.shape[0]

plt.xticks(np.arange(0, x_max, step))
plt.yticks(np.arange(0, y_max, step))
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, color='white')
plt.tick_params(axis='both', which='both',
                bottom=True, top=True, left=True, right=True,
                labelbottom=True, labeltop=False,
                labelleft=True, labelright=False)

# Добавляем подписи осей
plt.xlabel('Ширина (пиксели)', fontsize=10)
plt.ylabel('Высота (пиксели)', fontsize=10)
plt.title('Исходное изображение с размерной сеткой', fontsize=12)

# Правая панель - гистограмма со статистикой
ax2 = plt.subplot(1, 2, 2)
n, bins, patches = plt.hist(radii, bins=20, edgecolor='black', alpha=0.7, color='steelblue')

# Настраиваем гистограмму
plt.xlabel('Радиус (пиксели)', fontsize=12)
plt.ylabel('Количество капель', fontsize=12)
plt.title('Распределение капель по размерам', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Вычисляем статистику
total_drops = len(radii)
mean_radius = np.mean(radii) if radii else 0
median_radius = np.median(radii) if radii else 0

# Добавляем аннотацию
stats_text = (f'Всего капель: {total_drops}\n'
             f'Средний радиус: {mean_radius:.1f} px\n'
             f'Медианный радиус: {median_radius:.1f} px')

plt.text(0.95, 0.95, stats_text,
         transform=ax2.transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8,
                    edgecolor='gray', boxstyle='round'))

# Настраиваем общий вид
plt.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
plt.subplots_adjust(top=0.85)  # Регулировка верхнего отступа

# Добавляем общий заголовок
plt.suptitle('Анализ распределения капель', fontsize=16, y=0.98)

plt.show()