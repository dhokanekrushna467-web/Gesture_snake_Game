import pygame
import cv2
import mediapipe as mp
import random
import sys
import math
import numpy as np

# ------------------- Constants -------------------
FPS = 20
GRID_SIZE = 18
SMOOTHING = 0.18
BODY_DELAY = 6
MAX_PATH = 10000

# Colors
WHITE = (255, 255, 255)
GREEN_HEAD = (50, 220, 50)
GREEN_BODY = (0, 180, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# ------------------- Setup -------------------
def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    pygame.display.set_caption("Snake Tube - Hand + Arrow Control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    return screen, clock, font

def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    return cap

def get_mediapipe_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    return hands, mp_hands

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def snake_touch_food_any(snake_body, food_pos):
    fx, fy = food_pos
    for seg in snake_body:
        if dist(seg, [fx, fy]) < GRID_SIZE: return True
    return False

def get_tube_points(path, radius):
    points = []
    for i in range(len(path)-1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        dx, dy = x2 - x1, y2 - y1
        angle = math.atan2(dy, dx) + math.pi/2
        lx = x1 + radius * math.cos(angle)
        ly = y1 + radius * math.sin(angle)
        rx = x1 - radius * math.cos(angle)
        ry = y1 - radius * math.sin(angle)
        points.append(((lx, ly), (rx, ry)))
    return points

def draw_snake_tube(surface, snake_body):
    if len(snake_body) < 2: 
        pygame.draw.circle(surface, GREEN_BODY, (int(snake_body[0][0]), int(snake_body[0][1])), GRID_SIZE)
        return

    tube_points = get_tube_points(snake_body, GRID_SIZE)
    if len(tube_points) < 2:
        for seg in snake_body:
            pygame.draw.circle(surface, GREEN_BODY, (int(seg[0]), int(seg[1])), GRID_SIZE)
        return

    left_pts = [p[0] for p in tube_points]
    right_pts = [p[1] for p in tube_points][::-1]
    polygon = left_pts + right_pts
    if len(polygon) >= 3:
        pygame.draw.polygon(surface, GREEN_BODY, polygon, 0)

    hx, hy = snake_body[0]
    pygame.draw.circle(surface, GREEN_HEAD, (int(hx), int(hy)), GRID_SIZE+3)

def game_over_screen(screen, font, score, WIDTH, HEIGHT):
    while True:
        screen.fill(BLACK)
        t = font.render(f"Game Over! Score: {score}", True, WHITE)
        screen.blit(t, (WIDTH//2 - t.get_width()//2, HEIGHT//2 - 60))
        r1 = pygame.Rect(WIDTH//2 - 120, HEIGHT//2 + 10, 110, 50)
        r2 = pygame.Rect(WIDTH//2 + 10, HEIGHT//2 + 10, 110, 50)
        pygame.draw.rect(screen, GREEN_HEAD, r1)
        pygame.draw.rect(screen, RED, r2)
        screen.blit(font.render("Retry", True, BLACK), (r1.x+20, r1.y+10))
        screen.blit(font.render("Exit", True, BLACK), (r2.x+30, r2.y+10))
        pygame.display.update()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.MOUSEBUTTONDOWN:
                if r1.collidepoint(e.pos): return True
                if r2.collidepoint(e.pos): pygame.quit(); sys.exit()

def spawn_food(WIDTH, HEIGHT):
    while True:
        fx = random.randrange(GRID_SIZE, WIDTH - GRID_SIZE, GRID_SIZE)
        fy = random.randrange(GRID_SIZE, HEIGHT - GRID_SIZE, GRID_SIZE)
        if dist([fx, fy], [WIDTH//2, HEIGHT//2]) > 80: 
            return [fx, fy]

def process_hand_input(frame, hands, mp_hands, finger_x, finger_y, WIDTH, HEIGHT):
    try:
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            raw_x = int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH)
            raw_y = int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT)
            finger_x += SMOOTHING * (raw_x - finger_x)
            finger_y += SMOOTHING * (raw_y - finger_y)
            h, w, _ = frame.shape
            cv2.circle(frame, (int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                               int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)),
                       8, (0,0,255), -1)
            return finger_x, finger_y, frame, True
        return finger_x, finger_y, frame, False
    except:
        return finger_x, finger_y, frame, False


def main():
    screen, clock, font = init_pygame()
    WIDTH, HEIGHT = screen.get_width(), screen.get_height()
    hands, mp_hands = get_mediapipe_hands()
    cap = init_camera()

    path = [[WIDTH//2, HEIGHT//2]]
    snake_length = 1
    score = 0
    finger_x, finger_y = WIDTH//2, HEIGHT//2
    snake_body = [[WIDTH//2, HEIGHT//2]]
    food = spawn_food(WIDTH, HEIGHT)
    running = True
    paused = False

    # Arrow key control direction
    direction = [0, 0]  # dx, dy

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                cap.release(); pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    cap.release(); pygame.quit(); sys.exit()
                elif ev.key == pygame.K_p:
                    paused = not paused
                elif ev.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif ev.key == pygame.K_UP:
                    direction = [0, -1]
                elif ev.key == pygame.K_DOWN:
                    direction = [0, 1]
                elif ev.key == pygame.K_LEFT:
                    direction = [-1, 0]
                elif ev.key == pygame.K_RIGHT:
                    direction = [1, 0]

        if paused:
            screen.blit(font.render("Paused", True, WHITE), (WIDTH//2 - 80, HEIGHT//2))
            pygame.display.update()
            clock.tick(FPS)
            continue

        ret, frame = cap.read()
        if not ret: continue

        # Hand detection + fallback to arrow control
        finger_x, finger_y, frame, hand_found = process_hand_input(frame, hands, mp_hands, finger_x, finger_y, WIDTH, HEIGHT)

        hx, hy = snake_body[0]
        if hand_found:
            # Smoothly follow finger
            dx, dy = finger_x - hx, finger_y - hy
            d = math.hypot(dx, dy)
            if d > 1:
                step = min(GRID_SIZE, d)
                hx += step * dx / d
                hy += step * dy / d
        else:
            # Move using arrow keys
            hx += direction[0] * GRID_SIZE
            hy += direction[1] * GRID_SIZE

        path.insert(0, [hx, hy])
        if len(path) > MAX_PATH: path = path[:MAX_PATH]
        snake_body = path[::BODY_DELAY][:snake_length*2]

        if snake_touch_food_any(snake_body, food):
            score += 1
            snake_length += 1
            food = spawn_food(WIDTH, HEIGHT)

        for seg in snake_body[3:]:
            if dist(snake_body[0], seg) < GRID_SIZE * 0.8:
                cap.release()
                if game_over_screen(screen, font, score, WIDTH, HEIGHT): 
                    return main()
        if hx < 0 or hx > WIDTH or hy < 0 or hy > HEIGHT:
            cap.release()
            if game_over_screen(screen, font, score, WIDTH, HEIGHT): 
                return main()

        screen.fill(BLACK)
        draw_snake_tube(screen, snake_body)
        pygame.draw.rect(screen, RED, pygame.Rect(food[0], food[1], GRID_SIZE, GRID_SIZE))

        try:
            fs = cv2.resize(frame, (340, 260))
            fs = cv2.cvtColor(fs, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.rot90(fs))
            screen.blit(surf, (WIDTH - 360, 20))
        except: pass

        score_surf = font.render(f"Score: {score}", True, WHITE)
        inst_surf = font.render("ESC: Quit  P: Pause  F: Fullscreen  ↑↓←→: Move (if no hand)", True, WHITE)
        screen.blit(score_surf, (20, 20))
        screen.blit(inst_surf, (20, 60))

        pygame.display.update()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

