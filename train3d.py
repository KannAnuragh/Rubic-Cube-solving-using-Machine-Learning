# These are like importing tools we need to build our program
import raylibpy as pr    # This helps us draw 3D graphics and windows
import numpy as np       # This helps us do math with numbers and arrays
import queue             # For threaded solver communication
import configs           # Window size, FPS, camera settings
from rubik import Rubik  # Our 3D Rubik's cube
from cv_solver import RubikCVSolver  # Camera-based colour capture
from golden_solver import GoldenSolver  # Kociemba optimal solver


# ── Helper: map a move string to 3D rotation parameters ───────────────────
def map_move_to_game_input(move: str):
    """
    Convert a Kociemba move string into (axis, level, clockwise) tuples.

    "R"  → one clockwise rotation of the right face
    "R'" → one counter-clockwise rotation of the right face
    "R2" → two clockwise 90° rotations (= 180°) — NOTE: expand_moves()
           in golden_solver.py already splits R2 into ["R","R"] before
           this function is ever called, so you won't normally see "2" here.
    """
    clockwise = True
    times = 1

    if move.endswith("'"):
        clockwise = False
        move = move[:-1]       # strip the apostrophe
    elif move.endswith("2"):
        times = 2
        move = move[:-1]       # strip the "2"

    axis_map = {
        'U': (np.array([0, 1, 0]), 2),   # Up    → Y-axis, top level
        'D': (np.array([0, 1, 0]), 0),   # Down  → Y-axis, bottom level
        'R': (np.array([1, 0, 0]), 2),   # Right → X-axis, right level
        'L': (np.array([1, 0, 0]), 0),   # Left  → X-axis, left level
        'F': (np.array([0, 0, 1]), 2),   # Front → Z-axis, front level
        'B': (np.array([0, 0, 1]), 0),   # Back  → Z-axis, back level
    }

    result = axis_map.get(move)
    if result is None:
        return []
    axis, level = result
    return [(axis, level, clockwise)] * times


# ── Initialise objects ─────────────────────────────────────────────────────
pr.init_window(configs.window_w, configs.window_h, "Rubik's Cube  ·  Golden Solver")
pr.set_target_fps(configs.fps)

rubik_cube  = Rubik()
cv_solver   = RubikCVSolver()
golden      = GoldenSolver()       # Kociemba engine

rotation_queue = []

# State flags
solver_mode       = False   # Are we in CV-solver mode?
auto_solve        = False   # Is auto-solve running?
solution_ready    = False   # Do we have a solution queued up?
capture_completed = False   # Did we scan all 6 faces?

# ── Try to open the webcam ─────────────────────────────────────────────────
try:
    cv_solver.initialize_camera()
    camera_available = True
    print("✅ Camera ready.")
except Exception as e:
    camera_available = False
    print(f"⚠️  Camera not available: {e}")

# Loading splash
pr.begin_drawing()
pr.clear_background(pr.RAYWHITE)
pr.draw_text(b"Rubik's Cube  -  Golden Solver", 10, 10, 24, pr.DARKGRAY)
pr.draw_text(b"Initialising...", 10, 50, 16, pr.DARKGRAY)
pr.end_drawing()
pr.wait_time(0.1)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════
while not pr.window_should_close():

    shift_held = (pr.is_key_down(pr.KEY_LEFT_SHIFT) or
                  pr.is_key_down(pr.KEY_RIGHT_SHIFT))

    # ── Poll Thistlethwaite background thread ──────────────────────────────
    if cv_solver.is_solving:
        try:
            solution = cv_solver.solution_queue.get_nowait()
            if solution:
                cv_solver.solution_moves = [(m, f"Move {m}") for m in solution]
                solution_ready = True
                print("✅ Solver finished. Press SPACE to step or A to auto-solve.")
            else:
                print("❌ Solver found no solution.")
            cv_solver.is_solving = False
        except queue.Empty:
            pass   # still running — check again next frame

    # ── Keyboard input ─────────────────────────────────────────────────────
    if not cv_solver.is_solving:

        # ── Manual face rotations (F R U L B D keys) ──────────────────────
        if pr.is_key_pressed(pr.KEY_F):
            rotation_queue = rubik_cube.add_rotation(
                rotation_queue, np.array([0, 0, 1]), 2, not shift_held)

        elif pr.is_key_pressed(pr.KEY_R):
            rotation_queue = rubik_cube.add_rotation(
                rotation_queue, np.array([1, 0, 0]), 2, not shift_held)

        elif pr.is_key_pressed(pr.KEY_U):
            rotation_queue = rubik_cube.add_rotation(
                rotation_queue, np.array([0, 1, 0]), 2, not shift_held)

        elif pr.is_key_pressed(pr.KEY_L):
            rotation_queue = rubik_cube.add_rotation(
                rotation_queue, np.array([1, 0, 0]), 0, not shift_held)

        elif pr.is_key_pressed(pr.KEY_B):
            rotation_queue = rubik_cube.add_rotation(
                rotation_queue, np.array([0, 0, 1]), 0, not shift_held)

        elif pr.is_key_pressed(pr.KEY_D):
            rotation_queue = rubik_cube.add_rotation(
                rotation_queue, np.array([0, 1, 0]), 0, not shift_held)

        # ── C  →  Capture cube faces with camera ──────────────────────────
        elif pr.is_key_pressed(pr.KEY_C) and camera_available:
            print("📷 Starting face capture  (press 1–6 for each face, Q when done)…")
            cv_solver.capture_cube_state()
            if cv_solver.cube_state and len(cv_solver.cube_state) > 0:
                capture_completed = True
                # 🔑 IMPORTANT: DO NOT set solver_mode or update_colors here
                # User must press ENTER to apply the captured state
                num_faces = len(cv_solver.cube_state)
                print(f"✅ {num_faces} face(s) captured. Press ENTER to apply to cube.")
            else:
                print("⚠️  No faces captured — scan at least 1 face.")

        # ── G  →  GOLDEN SOLVE (Kociemba, ≤ 20 moves) ────────────────────
        elif pr.is_key_pressed(pr.KEY_G) and capture_completed:
            print("\n🏆 Golden Solver (Kociemba) running…")
            moves = golden.solve(cv_solver.cube_state, verbose=True)

            if moves:
                # Store as solution_moves so SPACE / A can step through them
                cv_solver.solution_moves = [(m, f"Golden {m}") for m in moves]
                cv_solver.current_step   = 0
                solution_ready = True
                solver_mode    = True
                print(f"✅ {len(moves)} moves queued. Press SPACE to step or A to auto-solve.")
            else:
                print("❌ Golden solve failed. Re-scan the cube and try again.")

        # ── T  →  Thistlethwaite (background thread, more moves) ──────────
        elif pr.is_key_pressed(pr.KEY_T) and solver_mode and capture_completed:
            print("🔄 Thistlethwaite solver starting…")
            if cv_solver.solve_cube(use_thistlethwaite=True):
                print("   Running in background. Wait for 'Solver finished' message.")
            else:
                print("❌ Thistlethwaite failed.")

        # ── S  →  Legacy solver ────────────────────────────────────────────
        elif pr.is_key_pressed(pr.KEY_S) and solver_mode and capture_completed:
            print("🔄 Legacy solver running…")
            if cv_solver.solve_cube(use_legacy=True):
                solution_ready = True
                print("✅ Legacy solution ready.")

        # ── SPACE  →  Execute next move in solution ────────────────────────
        elif pr.is_key_pressed(pr.KEY_SPACE) and solution_ready:
            move_data = cv_solver.get_current_move()
            if move_data and move_data[0]:
                for axis, level, cw in map_move_to_game_input(move_data[0]):
                    rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, level, cw)
                if not cv_solver.next_step():
                    print("✅ Solution complete!")
                    solution_ready = False
                    solver_mode    = False
            else:
                print("No more moves.")
                solution_ready = False

        # ── A  →  Toggle auto-solve ────────────────────────────────────────
        elif pr.is_key_pressed(pr.KEY_A) and solution_ready:
            auto_solve = not auto_solve
            print(f"Auto-solve: {'ON' if auto_solve else 'OFF'}")

        # ── ENTER  →  Apply captured cube state to 3D model ─────────────────
        elif pr.is_key_pressed(pr.KEY_ENTER):
            if cv_solver.cube_state is not None and len(cv_solver.cube_state) > 0:
                print("🎯 Applying captured cube state to 3D model...")
                # Debug: print the actual state
                print(f"Cube state: {cv_solver.cube_state}")
                # Apply colors to the 3D cube
                rubik_cube.update_colors(cv_solver.cube_state)
                # Reset capture for next round
                capture_completed = False
                print("✅ Cube updated successfully.")
            else:
                print("❌ No cube state available. Capture first (press C).")

        # ── ESC  →  Reset everything ───────────────────────────────────────
        elif pr.is_key_pressed(pr.KEY_ESCAPE):
            solution_ready    = False
            solver_mode       = False
            auto_solve        = False
            capture_completed = False
            cv_solver.current_step   = 0
            cv_solver.solution_moves = []
            print("🔄 Reset.")

    # ── Auto-solve: pump one move per frame (when not mid-rotation) ────────
    if auto_solve and solution_ready and not rubik_cube.is_rotating:
        move_data = cv_solver.get_current_move()
        if move_data and move_data[0]:
            for axis, level, cw in map_move_to_game_input(move_data[0]):
                rotation_queue = rubik_cube.add_rotation(rotation_queue, axis, level, cw)
            if not cv_solver.next_step():
                print("✅ Auto-solve complete!")
                auto_solve     = False
                solution_ready = False
                solver_mode    = False
        else:
            auto_solve     = False
            solution_ready = False

    # ── Animate rotations ──────────────────────────────────────────────────
    rotation_queue, _ = rubik_cube.handle_rotation(rotation_queue)

    # ── Camera orbit ───────────────────────────────────────────────────────
    pr.update_camera(configs.camera, pr.CAMERA_THIRD_PERSON)

    # ── Draw ───────────────────────────────────────────────────────────────
    pr.begin_drawing()
    pr.clear_background(pr.RAYWHITE)

    pr.begin_mode3d(configs.camera)
    pr.draw_grid(20, 1.0)
    for cube_group in rubik_cube.cubes:
        for cube_part in cube_group:
            pr.draw_model(cube_part.model, pr.Vector3(0, 0, 0), 1.0, pr.WHITE)
            pr.draw_model_wires(cube_part.model, pr.Vector3(0, 0, 0), 1.0, pr.DARKGRAY)
    pr.end_mode3d()

    # ── HUD ────────────────────────────────────────────────────────────────
    y = 10

    pr.draw_text(b"Manual: R L U D F B  (+SHIFT=CCW)", 10, y, 12, pr.DARKGRAY)
    y += 18

    pr.draw_text(b"C-Capture  G-Golden Solve  T-Thistlethwaite  S-Legacy", 10, y, 12, pr.DARKGRAY)
    y += 18

    pr.draw_text(b"SPACE-Step  A-Auto  ESC-Reset", 10, y, 12, pr.DARKGRAY)
    y += 24

    # Golden solver availability
    if golden._kociemba_available:
        pr.draw_text(b"G = Golden Solve  (Kociemba, <=20 moves)", 10, y, 12, pr.DARKGREEN)
    else:
        pr.draw_text(b"Golden Solver OFFLINE  (pip install kociemba)", 10, y, 12, pr.RED)
    y += 18

    # Capture status
    if capture_completed and cv_solver.cube_state:
        num_captured = len(cv_solver.cube_state)
        status = f"Faces: {num_captured}/6  ->  Press ENTER to apply".encode()
        pr.draw_text(status, 10, y, 12, pr.DARKGREEN)
    else:
        pr.draw_text(b"Faces: 0/6  (press C)", 10, y, 12, pr.GRAY)
    y += 18



    # Solver status
    if cv_solver.is_solving:
        pr.draw_text(b"Solving... please wait", 10, y, 14, pr.ORANGE)
        y += 20

    if solution_ready:
        move_data = cv_solver.get_current_move()
        if move_data and move_data[0]:
            label = f"Next move: {move_data[0]}".encode()
            pr.draw_text(label, 10, y, 14, pr.DARKBLUE)
            y += 18

        step, total = cv_solver.get_progress()
        prog = f"Progress: {step} / {total}".encode()
        pr.draw_text(prog, 10, y, 13, pr.DARKBLUE)
        y += 18

        if auto_solve:
            pr.draw_text(b"AUTO-SOLVE  ON", 10, y, 14, pr.ORANGE)
            y += 18

    if rubik_cube.is_rotating:
        pr.draw_text(b"Rotating...", 10, y, 14, pr.RED)
        y += 18

    if shift_held:
        pr.draw_text(b"SHIFT  Counter-Clockwise", 10, y, 13, pr.BLUE)

    pr.end_drawing()


# ── Cleanup ─────────────────────────────────────────────────────────────────
if camera_available:
    cv_solver.release_camera()
pr.close_window()