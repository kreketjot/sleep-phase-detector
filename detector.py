import cv2
import face_recognition
import numpy as np
import time
from callbacks import alarm_callback, sleep_callback  # Импорт коллбэков

# Загрузка известных лиц и их кодирование
known_face_encodings = []
known_face_names = []
unknown_counter = 1

def load_known_faces(face_files, face_names):
    global known_face_encodings, known_face_names
    for face_file, name in zip(face_files, face_names):
        image = face_recognition.load_image_file(face_file)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
        else:
            print(f"Warning: No faces found in the image {face_file}")

# Инициализация видеозахвата
video_capture = cv2.VideoCapture('./videos/head-pose-face-detection-female-and-male.mp4')

# Загрузка известных лиц
load_known_faces(["./faces/John.png", "./faces/Kate.png"], ["John", "Kate"])

# Инициализация переменных
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
previous_frame = None
trackers = {}
moving_persons = set()
lost_faces = {}
movement_start_time = {}
last_movement_time = {}
sleep_start_time = {}
alarm_threshold = 60  # 60 секунд
movement_timeout = 30  # 30 секунд
sleep_threshold = 900  # 15 минут
possible_movement_interval = 5  # 5 секунд

# Функция для получения следующего доступного имени для неизвестного лица
def get_next_unknown_name():
    global unknown_counter
    name = f"Unknown-{unknown_counter}"
    unknown_counter += 1
    return name

while True:
    # Захват одного кадра видео
    ret, frame = video_capture.read()
    if not ret:
        break

    # Детекция движения
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_frame is None:
        previous_frame = gray_frame
        continue

    # Вычисление абсолютной разницы между текущим и предыдущим кадром
    frame_diff = cv2.absdiff(previous_frame, gray_frame)
    previous_frame = gray_frame

    # Применение порогового значения для получения бинарного изображения
    _, thresh_frame = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    
    # Создание ядра для дилатации
    kernel = np.ones((5, 5), np.uint8)
    dilated_frame = cv2.dilate(thresh_frame, kernel, iterations=2)

    # Поиск контуров в бинарном изображении
    contours, _ = cv2.findContours(dilated_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    moving_persons.clear()
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Уменьшено для большей чувствительности
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        for name, tracker in trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                tx, ty, tw, th = map(int, bbox)
                if (tx <= x <= tx+tw) and (ty <= y <= ty+th):
                    moving_persons.add(name)

    # Обработка каждого второго кадра для экономии времени
    if process_this_frame:
        # Поиск всех лиц и их кодирование в текущем кадре видео
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        current_names = set(trackers.keys())
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Проверка, является ли лицо совпадающим с известным лицом
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = None

            # Использование известного лица с наименьшим расстоянием до нового лица
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                else:
                    name = get_next_unknown_name()
                    # Добавление нового неизвестного лица в известные лица
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)

            # Добавление трекера только если его еще нет
            if name not in trackers:
                face_names.append(name)
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (left, top, right-left, bottom-top))
                trackers[name] = tracker
                if name in lost_faces:
                    del lost_faces[name]

    process_this_frame = not process_this_frame

    # Обновление трекеров и отрисовка результатов
    current_names = set(trackers.keys())
    for name, tracker in list(trackers.items()):
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            overlay = frame.copy()
            color = (0, 0, 255) if name in moving_persons else (0, 255, 0)  # Красный для движущихся, зеленый для неподвижных
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            alpha = 0.3  # Прозрачность
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            if name:
                cv2.putText(frame, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
                
            # Обновление времени начала движения или его сброс
            if name in moving_persons:
                last_movement_time[name] = time.time()
                if name in movement_start_time:
                    if time.time() - movement_start_time[name] >= alarm_threshold:
                        alarm_callback(name)
                        del movement_start_time[name]  # Сбрасываем таймер после вызова будильника
                else:
                    movement_start_time[name] = time.time()
                if name in sleep_start_time:
                    del sleep_start_time[name]
            else:
                if name in movement_start_time and time.time() - last_movement_time[name] > movement_timeout:
                    del movement_start_time[name]
                    del last_movement_time[name]
                if name not in sleep_start_time:
                    sleep_start_time[name] = time.time()
                elif time.time() - sleep_start_time[name] >= sleep_threshold:
                    sleep_callback(name)
                    del sleep_start_time[name]
        else:
            print(f'LOST {name}')
            lost_faces[name] = bbox
            del trackers[name]

    # Отображение потерянных лиц
    for name, (lx, ly, lw, lh) in lost_faces.items():
        if name not in current_names:
            overlay = frame.copy()
            cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), -1)  # Голубой прямоугольник
            alpha = 0.1  # Прозрачность
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Отображение кадра с результатами
    cv2.imshow('Video', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение видеозахвата и закрытие всех окон
video_capture.release()
cv2.destroyAllWindows()