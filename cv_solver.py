import cv2
import numpy as np

class RubikCVSolver:
    def __init__(self):
        self.cube_state = None
        self.solution_moves = []
        self.current_step = 0
        self.is_solved = False
        self.camera = None

        self.color_ranges = {
            'white': ([0, 0, 180], [180, 60, 255]),
            'yellow': ([21, 100, 100], [35, 255, 255]),
            'red': ([0, 100, 100], [2, 200, 200]),
            'orange': ([1, 50, 50], [15, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'blue': ([50, 50, 50], [130, 255, 255])
        }

        self.face_names = ['F', 'B', 'R', 'L', 'U', 'D']

    def initialize_camera(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise Exception("Could not open camera")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def capture_cube_state(self):
        if not self.camera:
            raise Exception("Camera not initialized")

        print("=== Rubik's Cube Manual Capture ===")
        print("Align cube face to center rectangle and press 's' to capture")
        print("Press 'q' to quit at any time.")

        captured_faces = {}
        face_instructions = [
            "Show FRONT face",
            "Show RIGHT face", 
            "Show BACK face",
            "Show LEFT face",
            "Show TOP face",
            "Show BOTTOM face"
        ]

        current_face_idx = 0

        while current_face_idx < 6:
            ret, frame = self.camera.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            rect_size = 200
            x = center_x - rect_size // 2
            y = center_y - rect_size // 2

            # UI
            cv2.putText(display_frame, face_instructions[current_face_idx],
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Face {current_face_idx + 1}/6",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 's' to capture, 'q' to quit",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Draw box
            cv2.rectangle(display_frame,
                          (x, y), (x + rect_size, y + rect_size),
                          (0, 255, 255), 2)

            cv2.imshow("Rubik Cube Capture", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                face_info = {'bbox': (x, y, rect_size, rect_size)}
                colors = self.extract_colors_from_face(frame, face_info)
                if colors:
                    face_name = self.face_names[current_face_idx]
                    captured_faces[face_name] = colors
                    print(f"✓ Captured {face_name}:")
                    for row in colors:
                        print(f"  {row}")
                    current_face_idx += 1
                    cv2.putText(display_frame, "CAPTURED!",
                                (center_x - 50, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Rubik Cube Capture', display_frame)
                    cv2.waitKey(800)
            elif key == ord('q'):
                print("✗ Capture cancelled")
                break

        cv2.destroyAllWindows()

        if len(captured_faces) == 6:
            self.cube_state = captured_faces
            print("✓ All 6 faces captured manually!")
            return True
        else:
            print(f"✗ Only {len(captured_faces)} faces captured.")
            return False

    def extract_colors_from_face(self, frame, face_info):
        x, y, w, h = face_info['bbox']
        face_roi = frame[y:y + h, x:x + w]
        if face_roi.size == 0:
            return None
        face_roi = cv2.resize(face_roi, (300, 300))
        colors = []
        cell_size = 100
        for row in range(3):
            row_colors = []
            for col in range(3):
                cell = face_roi[row * cell_size + 20:(row + 1) * cell_size - 20,
                                col * cell_size + 20:(col + 1) * cell_size - 20]
                row_colors.append(self.get_dominant_color(cell))
            colors.append(row_colors)
        return colors

    def get_dominant_color(self, image_section):
        hsv = cv2.cvtColor(image_section, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        center = hsv[h // 3:2 * h // 3, w // 3:2 * w // 3]
        total_pixels = center.shape[0] * center.shape[1]

        best_match = 'unknown'
        max_ratio = 0

        for color, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)

            if color == 'red':
                mask1 = cv2.inRange(center, np.array([0, 120, 70]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(center, np.array([170, 120, 70]), np.array([180, 255, 255]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(center, lower, upper)

            match = cv2.countNonZero(mask) / total_pixels
            if match > max_ratio and match > 0.3:
                max_ratio = match
                best_match = color
        return best_match

    def solve_cube(self):
        self.solution_moves = [('R', 'Move R'), ('U', 'Move U')]
        self.current_step = 0
        self.is_solved = False
        return True

    def get_current_move(self):
        if self.current_step >= len(self.solution_moves):
            return None, "Done"
        return self.solution_moves[self.current_step]

    def next_step(self):
        self.current_step += 1
        if self.current_step >= len(self.solution_moves):
            self.is_solved = True
            return False
        return True

    def reset_solution(self):
        self.current_step = 0
        self.is_solved = False

    def get_progress(self):
        return self.current_step, len(self.solution_moves)

    def map_move_to_game_input(self, move):
        return {
            'R': ('R', False),
            'U': ('U', False)
        }.get(move, ('R', False))

    def release_camera(self):
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()

def integrate_cv_solver_with_rubik(rubik_cube, cv_solver):
    if cv_solver.is_solved:
        return None
    move, _ = cv_solver.get_current_move()
    key, ccw = cv_solver.map_move_to_game_input(move)
    axis_map = {
        'R': (np.array([1, 0, 0]), 2),
        'U': (np.array([0, 1, 0]), 2)
    }
    if key in axis_map:
        axis, lvl = axis_map[key]
        return axis, lvl, not ccw
    return None


def recapture_faces(self):
    print("=== Recapture Mode ===")
    print("Press 1-6 to choose face to recapture (1=F, 2=R, 3=B, 4=L, 5=U, 6=D), q to quit")

    face_mapping = {
        ord('1'): 'F', ord('2'): 'R', ord('3'): 'B',
        ord('4'): 'L', ord('5'): 'U', ord('6'): 'D'
    }

    while True:
        ret, frame = self.camera.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        rect_size = 200
        x = center_x - rect_size // 2
        y = center_y - rect_size // 2

        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 1-6 to recapture that face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to exit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(display_frame, (x, y), (x + rect_size, y + rect_size), (0, 255, 255), 2)

        cv2.imshow("Recapture", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in face_mapping:
            print(f"Recapturing face {face_mapping[key]}")
            face_info = {'bbox': (x, y, rect_size, rect_size)}
            colors = self.extract_colors_from_face(frame, face_info)
            if colors:
                self.cube_state[face_mapping[key]] = colors
                print(f"✓ Recaptured {face_mapping[key]}:", colors)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return True
