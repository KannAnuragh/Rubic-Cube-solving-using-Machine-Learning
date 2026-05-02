import numpy as np
from typing import Optional, List, Dict

# ---------------------------------------------------------------------------
# COLOR → FACE LETTER MAPPING (MATCHES YOUR CUBE)
# ---------------------------------------------------------------------------
COLOR_TO_FACE = {
    'white':  'U',
    'yellow': 'D',
    'red':    'F',
    'orange': 'B',
    'blue':   'R',   # your cube
    'green':  'L',   # your cube
}


# ---------------------------------------------------------------------------
# 🔧 FACE REMAPPING BY CENTERS
# ---------------------------------------------------------------------------
def remap_faces_by_centers(cube_state: Dict[str, List[List[str]]]) -> Dict[str, List[List[str]]]:
    """
    Reassign faces based on center colors.
    This guarantees correct cube orientation by using center stickers as authoritative.

    Expected center -> face mapping:
        white  -> U
        yellow -> D
        red    -> F
        orange -> B
        blue   -> R
        green  -> L
    """
    center_to_face = {
        'white':  'U',
        'yellow': 'D',
        'red':    'F',
        'orange': 'B',
        'blue':   'R',
        'green':  'L',
    }

    new_state: Dict[str, List[List[str]]] = {}

    for face, grid in cube_state.items():
        if not grid or not grid[1] or len(grid[1]) < 2:
            raise ValueError(f"Invalid grid for face '{face}'")

        center = grid[1][1]
        if center is None:
            raise ValueError(f"Missing center color for face '{face}'")

        mapped_face = center_to_face.get(center.lower())
        if mapped_face is None:
            raise ValueError(f"Invalid center color: {center}")

        if mapped_face in new_state:
            raise ValueError(f"Duplicate center mapping for color '{center}' -> '{mapped_face}'")

        new_state[mapped_face] = grid

    return new_state


# ---------------------------------------------------------------------------
# CONVERT TO KOCIEMBA STRING
# ---------------------------------------------------------------------------
def cube_state_to_kociemba(cube_state: Dict[str, List[List[str]]]) -> str:
    face_order = ['U', 'R', 'F', 'D', 'L', 'B']
    result = []

    for face in face_order:
        grid = cube_state.get(face)
        if grid is None:
            raise ValueError(f"Missing face '{face}'")

        for row in grid:
            for color in row:
                letter = COLOR_TO_FACE.get(color.lower())
                if letter is None:
                    raise ValueError(f"Unknown color '{color}'")
                result.append(letter)

    if len(result) != 54:
        raise ValueError("Invalid cube: not 54 stickers")

    return ''.join(result)


# ---------------------------------------------------------------------------
# MAIN SOLVER
# ---------------------------------------------------------------------------
class GoldenSolver:

    def __init__(self):
        try:
            import kociemba
            self._kociemba_available = True
            print("✅ GoldenSolver: Kociemba ready (≤20 moves)")
        except ImportError:
            self._kociemba_available = False
            print("❌ Install kociemba: pip install kociemba")

    # ------------------------------------------------------------------

    def solve(self, cube_state: Dict, verbose=True) -> Optional[List[str]]:
        if not self._kociemba_available:
            return None

        import kociemba
        from collections import Counter

        try:
            # 🔥 KEY FIX: remap faces by their CENTER colors BEFORE conversion
            cube_state = remap_faces_by_centers(cube_state)

            # Debug: print centers for sanity check
            print("\nCenters:")
            for f in ['U', 'D', 'F', 'B', 'R', 'L']:
                try:
                    center = cube_state[f][1][1]
                except Exception:
                    center = 'MISSING'
                print(f, center)

            cube_str = cube_state_to_kociemba(cube_state)

        except Exception as e:
            print(f"❌ Conversion error: {e}")
            return None

        if verbose:
            print("\n🔍 Kociemba input:", cube_str)
            print("Counts:", Counter(cube_str))

        try:
            solution = kociemba.solve(cube_str)
        except Exception as e:
            print(f"❌ Kociemba error: {e}")
            return None

        moves = self.expand_moves(solution)

        if verbose:
            print("\n🏆 SOLUTION")
            print("Raw :", solution)
            print("Moves:", len(solution.split()))
            print("Expanded:", " ".join(moves))

        return moves

    # ------------------------------------------------------------------

    @staticmethod
    def expand_moves(solution_str: str) -> List[str]:
        result = []
        for token in solution_str.split():
            if token.endswith('2'):
                result.extend([token[0], token[0]])
            else:
                result.append(token)
        return result