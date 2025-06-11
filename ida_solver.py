# ida_solver.py
import copy

FACES = ['U', 'D', 'L', 'R', 'F', 'B']
MOVES = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]

def is_solved(state):
    for face in state:
        color = state[face][0][0]
        for row in state[face]:
            if any(c != color for c in row):
                return False
    return True

def heuristic(state):
    count = 0
    for face in state:
        expected = state[face][1][1]  # center color
        for row in state[face]:
            for c in row:
                if c != expected:
                    count += 1
    return count

def rotate_face(face_data, cw=True):
    if cw:
        return [list(reversed(col)) for col in zip(*face_data)]
    else:
        return list(map(list, zip(*face_data)))[::-1]

def apply_move(state, move):
    new_state = copy.deepcopy(state)

    if move in ["R", "R'"]:
        cw = move == "R"
        new_state['R'] = rotate_face(new_state['R'], cw=cw)
        for i in range(3):
            u, f, d, b = state['U'][i][2], state['F'][i][2], state['D'][i][2], state['B'][2 - i][0]
            if cw:
                new_state['F'][i][2] = u
                new_state['D'][i][2] = f
                new_state['B'][2 - i][0] = d
                new_state['U'][i][2] = b
            else:
                new_state['B'][2 - i][0] = u
                new_state['D'][i][2] = b
                new_state['F'][i][2] = d
                new_state['U'][i][2] = f

    elif move in ["L", "L'"]:
        cw = move == "L"
        new_state['L'] = rotate_face(new_state['L'], cw=cw)
        for i in range(3):
            u, f, d, b = state['U'][i][0], state['F'][i][0], state['D'][i][0], state['B'][2 - i][2]
            if cw:
                new_state['F'][i][0] = u
                new_state['D'][i][0] = f
                new_state['B'][2 - i][2] = d
                new_state['U'][i][0] = b
            else:
                new_state['B'][2 - i][2] = u
                new_state['D'][i][0] = b
                new_state['F'][i][0] = d
                new_state['U'][i][0] = f

    elif move in ["U", "U'"]:
        cw = move == "U"
        new_state['U'] = rotate_face(new_state['U'], cw=cw)
        if cw:
            temp = state['F'][0][:]
            new_state['F'][0] = state['R'][0][:]
            new_state['R'][0] = state['B'][0][:]
            new_state['B'][0] = state['L'][0][:]
            new_state['L'][0] = temp
        else:
            temp = state['F'][0][:]
            new_state['F'][0] = state['L'][0][:]
            new_state['L'][0] = state['B'][0][:]
            new_state['B'][0] = state['R'][0][:]
            new_state['R'][0] = temp

    elif move in ["D", "D'"]:
        cw = move == "D"
        new_state['D'] = rotate_face(new_state['D'], cw=cw)
        if cw:
            temp = state['F'][2][:]
            new_state['F'][2] = state['L'][2][:]
            new_state['L'][2] = state['B'][2][:]
            new_state['B'][2] = state['R'][2][:]
            new_state['R'][2] = temp
        else:
            temp = state['F'][2][:]
            new_state['F'][2] = state['R'][2][:]
            new_state['R'][2] = state['B'][2][:]
            new_state['B'][2] = state['L'][2][:]
            new_state['L'][2] = temp

    elif move in ["F", "F'"]:
        cw = move == "F"
        new_state['F'] = rotate_face(new_state['F'], cw=cw)
        for i in range(3):
            u, r, d, l = state['U'][2][i], state['R'][i][0], state['D'][0][2 - i], state['L'][2 - i][2]
            if cw:
                new_state['R'][i][0] = u
                new_state['D'][0][2 - i] = r
                new_state['L'][2 - i][2] = d
                new_state['U'][2][i] = l
            else:
                new_state['L'][2 - i][2] = u
                new_state['D'][0][2 - i] = l
                new_state['R'][i][0] = d
                new_state['U'][2][i] = r

    elif move in ["B", "B'"]:
        cw = move == "B"
        new_state['B'] = rotate_face(new_state['B'], cw=cw)
        for i in range(3):
            u, l, d, r = state['U'][0][2 - i], state['L'][i][0], state['D'][2][i], state['R'][2 - i][2]
            if cw:
                new_state['L'][i][0] = u
                new_state['D'][2][i] = l
                new_state['R'][2 - i][2] = d
                new_state['U'][0][2 - i] = r
            else:
                new_state['R'][2 - i][2] = u
                new_state['D'][2][i] = r
                new_state['L'][i][0] = d
                new_state['U'][0][2 - i] = l

    return new_state

def ida_star(initial_state, max_depth=10):
    threshold = heuristic(initial_state)
    path = [(initial_state, None)]

    def search(g, threshold):
        node = path[-1][0]
        f = g + heuristic(node)
        if f > threshold:
            return f
        if is_solved(node):
            return "FOUND"

        min_cost = float('inf')
        for move in MOVES:
            next_state = apply_move(node, move)
            if len(path) > 1 and next_state == path[-2][0]:
                continue
            path.append((next_state, move))
            t = search(g + 1, threshold)
            if t == "FOUND":
                return "FOUND"
            if t < min_cost:
                min_cost = t
            path.pop()
        return min_cost

    while True:
        t = search(0, threshold)
        if t == "FOUND":
            return [m for _, m in path[1:] if m]
        if t == float('inf') or threshold > max_depth:
            return None
        threshold = t
