import cv2
import numpy as np
import pickle
import os
from scipy.optimize import least_squares
from collections import defaultdict
from datetime import datetime
import json

class VideoFilter:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_pos = 0
        self.paused = False
        self.params_file = "filter_params.pkl"
        self.MIN_CONTOUR_LENGTH = 20
        self.output_dir = "output_results"
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ellipse_params_file = os.path.join(self.output_dir, f"ellipse_params_{timestamp}.json")

        self.default_params = {
            'T_min': 100,
            'T_max': 250,
            'd': 2,
            'sigma_color': 125,
            'sigma_space': 125,
            'min_contour_area': 10,
            'residuals_threshold': 1.0,
        }

        self.load_or_init_params()

        cv2.namedWindow('Properties', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
        self.init_trackbars()

    def load_or_init_params(self):
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'rb') as f:
                    loaded_params = pickle.load(f)
                    self.params = {**self.default_params, **loaded_params}
            except:
                self.params = self.default_params.copy()
        else:
            self.params = self.default_params.copy()

    def init_trackbars(self):
        for key in self.default_params:
            if key not in self.params:
                self.params[key] = self.default_params[key]

        cv2.createTrackbar('Frame', 'Properties', 0, self.total_frames - 1, self.set_frame)
        cv2.createTrackbar('T_min', 'Properties', self.params['T_min'], 255, lambda x: None)
        cv2.createTrackbar('T_max', 'Properties', self.params['T_max'], 255, lambda x: None)
        cv2.createTrackbar('d', 'Properties', self.params['d'], 20, lambda x: None)
        cv2.createTrackbar('sigma_color', 'Properties', self.params['sigma_color'], 200, lambda x: None)
        cv2.createTrackbar('sigma_space', 'Properties', self.params['sigma_space'], 200, lambda x: None)
        cv2.createTrackbar('min_area', 'Properties', self.params['min_contour_area'], 1000, lambda x: None)
        cv2.createTrackbar('residuals_th', 'Properties', int(self.params['residuals_threshold'] * 100), 500,
                           lambda x: None)

    def save_params(self):
        with open(self.params_file, 'wb') as f:
            pickle.dump(self.params, f)
        print("Параметры фильтра сохранены.")

    def set_frame(self, pos):
        self.current_frame_pos = pos
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def update_params(self):
        self.params = {
            'T_min': cv2.getTrackbarPos('T_min', 'Properties'),
            'T_max': cv2.getTrackbarPos('T_max', 'Properties'),
            'd': cv2.getTrackbarPos('d', 'Properties'),
            'sigma_color': cv2.getTrackbarPos('sigma_color', 'Properties'),
            'sigma_space': cv2.getTrackbarPos('sigma_space', 'Properties'),
            'min_contour_area': cv2.getTrackbarPos('min_area', 'Properties'),
            'residuals_threshold': cv2.getTrackbarPos('residuals_th', 'Properties') / 100.0,
        }
        return self.params

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray,
                                       d=self.params['d'],
                                       sigmaColor=self.params['sigma_color'],
                                       sigmaSpace=self.params['sigma_space'])
        edges = cv2.Canny(filtered,
                          self.params['T_min'],
                          self.params['T_max'])
        return edges

    def ellipse_residuals(self, params, x, y):
        xc, yc, a, b, theta = params
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        x_rot = (x - xc) * cos_t + (y - yc) * sin_t
        y_rot = (y - yc) * cos_t - (x - xc) * sin_t

        a = max(a, 1e-6)
        b = max(b, 1e-6)

        return (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1

    def fit_ellipse(self, xe_vec, ye_vec):
        x_mean, y_mean = np.mean(xe_vec), np.mean(ye_vec)
        a_init = (np.max(xe_vec) - np.min(xe_vec)) / 2
        b_init = (np.max(ye_vec) - np.min(ye_vec)) / 2
        theta_init = 0

        initial_params = [x_mean, y_mean, a_init, b_init, theta_init]

        result = least_squares(self.ellipse_residuals, initial_params, args=(xe_vec, ye_vec))

        xc, yc, a, b, theta = result.x
        residuals = result.fun

        return xc, yc, a, b, theta, residuals

    def process_contours(self, frame, edges, frame_num):
        binary_img = np.zeros_like(edges, dtype=np.uint8)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            result_img = frame.copy()
            cv2.putText(result_img, 'Droplets: 0', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return result_img, [], [], []

        cv2.drawContours(binary_img, contours, -1, 255, thickness=-1)
        kernel = np.ones((3, 3), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        final_contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        droplet_contours = []
        droplet_centers = []
        ellipse_params_list = []
        result_img = frame.copy()

        for cnt in final_contours:
            if len(cnt) < self.MIN_CONTOUR_LENGTH:
                continue

            cnt_area = cv2.contourArea(cnt)
            if cnt_area < self.params['min_contour_area']:
                continue

            if len(cnt) >= 5:
                try:
                    xe_vec = cnt[:, 0, 0]
                    ye_vec = cnt[:, 0, 1]
                    xc, yc, a, b, theta, residuals = self.fit_ellipse(xe_vec, ye_vec)

                    if min(a, b) > 0:
                        axis_ratio = max(a, b) / min(a, b)
                        if axis_ratio >= 100:
                            continue

                    rmse = np.sqrt(np.mean(residuals ** 2))
                    if rmse < self.params['residuals_threshold']:
                        droplet_contours.append(cnt)
                        droplet_centers.append((int(xc), int(yc)))
                        ellipse_params_list.append({
                            'frame': frame_num,
                            'center_x': float(xc),
                            'center_y': float(yc),
                            'semi_major_axis': float(max(a, b)),
                            'semi_minor_axis': float(min(a, b)),
                            'theta': float(np.degrees(theta)),
                            'rmse': float(rmse)
                        })
                        ellipse = ((int(xc), int(yc)), (int(2 * a), int(2 * b)), np.degrees(theta))
                        cv2.ellipse(result_img, ellipse, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Ошибка при подгонке эллипса: {e}")
                    continue

        cv2.putText(result_img, f'Droplets: {len(droplet_contours)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return result_img, droplet_contours, droplet_centers, ellipse_params_list

    def save_ellipse_params(self, all_ellipse_params):
        with open(self.ellipse_params_file, 'w') as f:
            json.dump(all_ellipse_params, f, indent=4)
        print(f"Параметры эллипсов сохранены в файл: {self.ellipse_params_file}")

    def process_all_frames(self):
        all_frame_data = [([], []) for _ in range(self.total_frames)]
        all_ellipse_params = []
        cap = cv2.VideoCapture(self.video_path)

        print("Обработка всех кадров для сегментации...")
        for frame_num in range(self.total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            edges = self.preprocess_frame(frame)
            _, droplet_contours, droplet_centers, ellipse_params_list = self.process_contours(frame, edges, frame_num)
            all_frame_data[frame_num] = (droplet_contours, droplet_centers)
            all_ellipse_params.extend(ellipse_params_list)

            if frame_num % 100 == 0:
                print(f"Обработано кадров для сегментации: {frame_num}/{self.total_frames}")

        print("Сегментация завершена!")
        cap.release()
        return all_frame_data, all_ellipse_params

    def run(self):
        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.current_frame_pos = 0
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cv2.setTrackbarPos('Frame', 'Properties', self.current_frame_pos)

                current_params = self.update_params()
                edges = self.preprocess_frame(frame)
                result_img, _, _, _ = self.process_contours(frame, edges, self.current_frame_pos)

                cv2.imshow('Segmentation', result_img)

                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    self.paused = not self.paused
                elif key == ord('q'):
                    self.save_params()
                    self.cap.release()
                    cv2.destroyAllWindows()
                    all_frame_data, all_ellipse_params = self.process_all_frames()
                    self.save_ellipse_params(all_ellipse_params)
                    return current_params, all_frame_data, all_ellipse_params
                elif key == ord('s'):
                    self.save_params()
                elif key == ord('r'):
                    self.current_frame_pos = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                elif key == ord('d'):
                    self.current_frame_pos = min(self.current_frame_pos + 10, self.total_frames - 1)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)
                elif key == ord('a'):
                    self.current_frame_pos = max(self.current_frame_pos - 10, 0)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

class DropletAnalyzer:
    def __init__(self, video_path, filter_params, all_frame_data, max_distance, max_frames_missing, all_ellipse_params):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.filter_params = filter_params
        self.all_frame_data = all_frame_data
        self.all_ellipse_params = all_ellipse_params
        self.MAX_DISTANCE = max_distance
        self.MAX_FRAMES_MISSING = max_frames_missing

        self.trajectories = defaultdict(list)
        self.next_id = 0
        self.updated_ellipse_params = []  # Список для параметров эллипсов с track_id

        cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)

        self.output_dir = "output_results"
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trajectories_file = os.path.join(self.output_dir, f"trajectories_{timestamp}.json")
        self.updated_ellipse_file = os.path.join(self.output_dir, f"updated_ellipse_params_{timestamp}.json")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video_file = os.path.join(self.output_dir, f"trajectories_{timestamp}.avi")
        self.out = cv2.VideoWriter(self.output_video_file, fourcc, self.fps, (self.frame_width, self.frame_height))
        print(f"Сохранение видео с траекториями в файл: {self.output_video_file}")

    def get_new_track_id(self):
        track_id = f"D-{self.next_id:03d}"
        self.next_id += 1
        return track_id

    def assign_ellipse_params(self, frame_num, droplet, track_id):
        """Сопоставляет параметры эллипса с каплями по координатам центра и добавляет track_id."""
        droplet_x, droplet_y = droplet
        min_dist = float('inf')
        best_match = None

        for param in self.all_ellipse_params:
            if param['frame'] != frame_num:
                continue

            dist = np.sqrt((param['center_x'] - droplet_x) ** 2 + (param['center_y'] - droplet_y) ** 2)
            if dist < min_dist and dist < self.MAX_DISTANCE:
                min_dist = dist
                best_match = param

        if best_match:
            updated_param = best_match.copy()
            updated_param['track_id'] = track_id
            self.updated_ellipse_params.append(updated_param)
            return updated_param
        return None

    def update_tracks(self, frame_droplets, frame_num):
        if not self.trajectories:
            for droplet in frame_droplets:
                track_id = self.get_new_track_id()
                ellipse_param = self.assign_ellipse_params(frame_num, droplet, track_id)
                if ellipse_param:
                    self.trajectories[track_id].append({
                        'frame': frame_num,
                        'center_x': ellipse_param['center_x'],
                        'center_y': ellipse_param['center_y']
                    })
            return

        matched_droplets = set()
        matched_tracks = set()

        for track_id, track in self.trajectories.items():
            if not track:
                continue

            last_point = track[-1]
            last_frame = last_point['frame']
            last_x, last_y = last_point['center_x'], last_point['center_y']

            if frame_num - last_frame > self.MAX_FRAMES_MISSING:
                continue

            min_dist = float('inf')
            best_match = None
            best_droplet = None

            for i, droplet in enumerate(frame_droplets):
                if i in matched_droplets:
                    continue

                dist = np.sqrt((droplet[0] - last_x) ** 2 + (droplet[1] - last_y) ** 2)

                if dist < self.MAX_DISTANCE and dist < min_dist:
                    min_dist = dist
                    best_match = i
                    best_droplet = droplet

            if best_match is not None:
                ellipse_param = self.assign_ellipse_params(frame_num, best_droplet, track_id)
                if ellipse_param:
                    self.trajectories[track_id].append({
                        'frame': frame_num,
                        'center_x': ellipse_param['center_x'],
                        'center_y': ellipse_param['center_y']
                    })
                    matched_droplets.add(best_match)
                    matched_tracks.add(track_id)

        for i, droplet in enumerate(frame_droplets):
            if i not in matched_droplets:
                new_track_id = self.get_new_track_id()
                ellipse_param = self.assign_ellipse_params(frame_num, droplet, new_track_id)
                if ellipse_param:
                    self.trajectories[new_track_id].append({
                        'frame': frame_num,
                        'center_x': ellipse_param['center_x'],
                        'center_y': ellipse_param['center_y']
                    })

    def draw_trajectories_and_ellipses(self, image, frame_num):
        # Отрисовка контуров
        contours, _ = self.all_frame_data[frame_num]
        for cnt in contours:
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

        # Отрисовка эллипсов и траекторий
        for track_id, track in self.trajectories.items():
            if not track:
                continue

            # Получаем точки траектории до текущего кадра
            points = [(point['center_x'], point['center_y']) for point in track if point['frame'] <= frame_num]

            # Отрисовка траектории (линии между центрами эллипсов)
            for i in range(1, len(points)):
                cv2.line(image,
                         (int(points[i - 1][0]), int(points[i - 1][1])),
                         (int(points[i][0]), int(points[i][1])),
                         (255, 0, 0), 2)

            # Отрисовка эллипса и метки track_id для текущего кадра
            for param in self.updated_ellipse_params:
                if param['frame'] == frame_num and param['track_id'] == track_id:
                    xc, yc = int(param['center_x']), int(param['center_y'])
                    a, b = param['semi_major_axis'], param['semi_minor_axis']
                    theta = param['theta']
                    ellipse = ((xc, yc), (int(2 * a), int(2 * b)), theta)
                    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
                    cv2.putText(image, str(track_id),
                                (xc, yc),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    break

        return image

    def analyze(self):
        print(f"Начало анализа видео: {self.video_path}")
        print(f"Всего кадров: {self.total_frames}")

        try:
            for frame_num in range(self.total_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break

                _, centers = self.all_frame_data[frame_num]
                self.update_tracks(centers, frame_num)
                trajectory_frame = self.draw_trajectories_and_ellipses(frame.copy(), frame_num)

                self.out.write(trajectory_frame)

                cv2.putText(trajectory_frame, f'Droplets: {len(centers)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(trajectory_frame, f'Tracks: {len(self.trajectories)}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow('Analysis', trajectory_frame)

                if frame_num % 100 == 0:
                    print(f"Обработано кадров: {frame_num}/{self.total_frames}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.save_trajectories()
            self.save_updated_ellipse_params()
            print(f"Обнаружено траекторий: {len(self.trajectories)}")
            self.out.release()
            print(f"Видео с траекториями сохранено в: {self.output_video_file}")
            print("Анализ завершен!")

    def save_trajectories(self):
        trajectories_dict = {}
        for track_id, points in self.trajectories.items():
            trajectories_dict[track_id] = [{
                'x': point['center_x'],
                'y': point['center_y'],
                'frame': point['frame']
            } for point in points]

        with open(self.trajectories_file, 'w') as f:
            json.dump(trajectories_dict, f, indent=4)

        print(f"Траектории сохранены в файл: {self.trajectories_file}")

    def save_updated_ellipse_params(self):
        with open(self.updated_ellipse_file, 'w') as f:
            json.dump(self.updated_ellipse_params, f, indent=4)
        print(f"Обновлённые параметры эллипсов сохранены в файл: {self.updated_ellipse_file}")

if __name__ == "__main__":
    video_path = r'D:\Pycharm\Project_MAIN\VKR_OpenCV\video\1123124.mp4'

    MAX_DISTANCE = 10
    MAX_FRAMES_MISSING = 10

    print("Настройте параметры фильтра и нажмите 'q' для продолжения...")
    filter = VideoFilter(video_path)
    filter_params, all_frame_data, all_ellipse_params = filter.run()

    if filter_params:
        print("\nЗапуск анализа с выбранными параметрами...")
        print(f"MAX_DISTANCE: {MAX_DISTANCE}, MAX_FRAMES_MISSING: {MAX_FRAMES_MISSING}")
        analyzer = DropletAnalyzer(video_path, filter_params, all_frame_data, MAX_DISTANCE, MAX_FRAMES_MISSING, all_ellipse_params)
        analyzer.analyze()