import heapq
from snake import Directions, TILE_SIZE, WIDTH, HEIGHT

GRID_W = WIDTH // TILE_SIZE
GRID_H = HEIGHT // TILE_SIZE

_CLOCK_WISE = [(-1, 0), (0, -1), (1, 0), (0, 1)]


def _to_grid(rect):
    return (rect.x // TILE_SIZE, rect.y // TILE_SIZE)


def _facing_to_action(current_facing, desired_facing):
    if desired_facing not in _CLOCK_WISE:
        return [1, 0, 0]
    cur_idx = _CLOCK_WISE.index(current_facing)
    des_idx = _CLOCK_WISE.index(desired_facing)
    diff = (des_idx - cur_idx) % 4
    if diff == 0:
        return [1, 0, 0]   # FORWARD
    elif diff == 3:         # CCW (LEFT action uses +3 mod 4)
        return [0, 1, 0]   # LEFT
    elif diff == 1:         # CW (RIGHT action uses +1 mod 4)
        return [0, 0, 1]   # RIGHT
    else:
        return [0, 1, 0]


def _astar(start, goal, obstacles):
    h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    # (f, g, position)
    open_heap = [(h(start), 0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_heap:
        f, g, cur = heapq.heappop(open_heap)

        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        if g > g_score.get(cur, float('inf')):
            continue

        for dx, dy in _CLOCK_WISE:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < GRID_W and 0 <= nb[1] < GRID_H):
                continue
            if nb in obstacles:
                continue
            new_g = g + 1
            if new_g < g_score.get(nb, float('inf')):
                came_from[nb] = cur
                g_score[nb] = new_g
                heapq.heappush(open_heap, (new_g + h(nb), new_g, nb))

    return None


def _flood_fill(start, obstacles):
    visited = {start}
    queue = [start]
    while queue:
        cur = queue.pop()
        for dx, dy in _CLOCK_WISE:
            nb = (cur[0] + dx, cur[1] + dy)
            if (0 <= nb[0] < GRID_W and 0 <= nb[1] < GRID_H
                    and nb not in obstacles and nb not in visited):
                visited.add(nb)
                queue.append(nb)
    return len(visited)


class AStarAgent:
    label = "A*"

    def __init__(self, pretrained=True):
        self.n_games = 0
        self.explore = False

    def get_action(self, state, game):
        snake = game.snake
        head = _to_grid(snake.body[0])
        apple = _to_grid(game.apple)

        body_obstacles = set(_to_grid(part) for part in snake.body[1:-1])

        path = _astar(head, apple, body_obstacles)

        if path:
            next_cell = path[0]
            desired_facing = (next_cell[0] - head[0], next_cell[1] - head[1])
            return _facing_to_action(snake.facing, desired_facing)

        best_action = None
        best_space = -1
        # Include full body (without tail) as obstacles for reachability check
        full_obstacles = set(_to_grid(part) for part in snake.body[1:])

        for action in [d.value for d in Directions]:
            desired_facing = snake.get_new_facing(action)
            next_head = snake.get_next_head(desired_facing)
            next_pos = (next_head.x // TILE_SIZE, next_head.y // TILE_SIZE)

            if snake.collision(next_head):
                continue

            space = _flood_fill(next_pos, full_obstacles)
            if space > best_space:
                best_space = space
                best_action = action

        return best_action if best_action is not None else [1, 0, 0]


class AStarDumbAgent:
    label = "A* Dumb"

    def __init__(self, pretrained=True):
        self.n_games = 0
        self.explore = False

    def get_action(self, state, game):
        snake = game.snake
        head = _to_grid(snake.body[0])
        apple = _to_grid(game.apple)

        body_obstacles = set(_to_grid(part) for part in snake.body[1:-1])
        path = _astar(head, apple, body_obstacles)

        if path:
            next_cell = path[0]
            desired_facing = (next_cell[0] - head[0], next_cell[1] - head[1])
            return _facing_to_action(snake.facing, desired_facing)

        # Dumb fallback: first non-colliding action, no lookahead
        for action in [d.value for d in Directions]:
            facing = snake.get_new_facing(action)
            if not snake.collision(snake.get_next_head(facing)):
                return action
        return [1, 0, 0]


class AStarVeryDumbAgent:
    label = "A* Very Dumb"

    def __init__(self, pretrained=True):
        self.n_games = 0
        self.explore = False

    def get_action(self, state, game):
        snake = game.snake
        head = _to_grid(snake.body[0])
        apple = _to_grid(game.apple)

        body_obstacles = set(_to_grid(part) for part in snake.body[1:-1])
        path = _astar(head, apple, body_obstacles)

        if path:
            next_cell = path[0]
            desired_facing = (next_cell[0] - head[0], next_cell[1] - head[1])
            return _facing_to_action(snake.facing, desired_facing)

        return [1, 0, 0]


if __name__ == "__main__":
    from snake import SnakeGame

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    high_score = 0
    game = SnakeGame()
    agent = AStarVeryDumbAgent()

    while True:
        state = game.get_game_state()
        action = agent.get_action(state, game)
        reward, game_over, score = game.play_step(action, agent)

        if game_over:
            agent.n_games += 1
            total_score += score
            mean_score = total_score / agent.n_games

            if score > high_score:
                high_score = score

            print("A* Game:", agent.n_games, "Score:", score, "Record:", high_score)

            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            game.restart()
