import random

import cv2
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import numpy as np


# Hand Connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

gesture      = {"type": "none"}   # "point", "open", "fist", "none"
direction    = {"x": 0, "y": 0}
cursor_pos   = {"x": 400, "y": 300}
camera_frame = {"surface": None}
frame_lock   = threading.Lock()

def count_extended_fingers(lms):
    tips    = [8, 12, 16, 20]
    middles = [6, 10, 14, 18]
    return sum(1 for t, m in zip(tips, middles) if lms[t].y < lms[m].y)

def get_gesture(lms):
    n = count_extended_fingers(lms)
    tip      = lms[8]
    mid_base = lms[9]
    if tip.y > mid_base.y + 0.05:
        return "point"
    if n == 0:
        return "fist"
    elif n <= 3:
        return "point"
    elif n >= 4:
        return "open"
    return "none"

def get_direction(lms, w, h):
    tip  = lms[8]
    base = lms[5]
    dx   = (tip.x - base.x) * w
    dy   = (tip.y - base.y) * h

    if abs(dx) < 10 and abs(dy) < 10:
        return 0, 0
    if abs(dx) > abs(dy) * 0.7:
        return (-1, 0) if dx > 0 else (1, 0)
    else:
        return (0, 1) if dy > 0 else (0, -1)

def get_cursor(lms, w, h, screen_w, screen_h):
    cx = (1.0 - lms[8].x) * screen_w
    cy = lms[8].y * screen_h
    return int(np.clip(cx, 0, screen_w)), int(np.clip(cy, 0, screen_h))

def draw_skeleton_on_black(shape, detection_result):
    h, w = shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if not detection_result.hand_landmarks:
        return canvas

    for hand_landmarks in detection_result.hand_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, points[a], points[b], (0, 255, 80), 2)
        for i, pt in enumerate(points):
            color = (0, 0, 255) if i in (4, 8, 12, 16, 20) else (0, 220, 220)
            cv2.circle(canvas, pt, 6, color, -1)

    return canvas

def hand_tracker_thread():
    MODEL_PATH = "hand_landmarker (1).task"
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap   = cv2.VideoCapture(0)
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts       = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result   = landmarker.detect_for_video(mp_image, ts)

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                g   = get_gesture(lms)
                gesture["type"] = g

                if g == "point":
                    dx, dy = get_direction(lms, cam_w, cam_h)
                    direction["x"] = dx
                    direction["y"] = dy
                elif g == "open":
                    cx, cy = get_cursor(lms, cam_w, cam_h, 800, 600)
                    cursor_pos["x"] = cx
                    cursor_pos["y"] = cy
                    direction["x"]  = 0
                    direction["y"]  = 0
                else:
                    direction["x"] = 0
                    direction["y"] = 0
            else:
                gesture["type"] = "none"
                direction["x"]  = 0
                direction["y"]  = 0

            skeleton = draw_skeleton_on_black((cam_h, cam_w), result)
            overlay_w, overlay_h = 240, 180
            resized  = cv2.resize(skeleton, (overlay_w, overlay_h))
            rgb_surf = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            surface  = pygame.surfarray.make_surface(np.rot90(rgb_surf))

            with frame_lock:
                camera_frame["surface"] = surface

    cap.release()


t = threading.Thread(target=hand_tracker_thread, daemon=True)
t.start()

# =============================================================Game Setup==============================================================
pygame.init()
SCREEN_W, SCREEN_H = 800, 600
SPEED               = 4
OVERLAY_W, OVERLAY_H = 240, 180
PAD                 = 10
SCROLL_EDGE         = SCREEN_W - 50

screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Gesture Block")

clock = pygame.time.Clock()
font  = pygame.font.SysFont("Minecraft.ttf", 28)
small = pygame.font.SysFont("Minecraft.ttf", 22)
large = pygame.font.SysFont("Minecraft.ttf", 35)

pygame.mixer.init()
pygame.mixer.music.load('song1.mp3')
pygame.mixer.music.play()

GESTURE_COLORS = {
    "point": (255, 220, 0),
    "open":  (0, 200, 255),
    "fist":  (255, 60, 60),
    "none":  (150, 150, 150),
}


# =============================================================Level 1==============================================================
def level_1():
    # Load assets
    background = pygame.image.load("Images/sky_background.png").convert()
    background = pygame.transform.scale(background, (SCREEN_W, SCREEN_H))
    sprite_img = pygame.image.load("Images/Lani_down.png").convert_alpha()
    sprite_img = pygame.transform.scale(sprite_img, (50, 50))
    floor_img = pygame.image.load("Images/floor.png").convert()
    floor_img = pygame.transform.scale(floor_img, (1600, 50))
    floor = floor_img.get_rect(topleft=(0, 550))

    duck_img = pygame.image.load("Images/duck.png").convert_alpha()
    duck_img = pygame.transform.scale(duck_img, (50, 50))
    plant_img = pygame.image.load("Images/plant.png").convert_alpha()
    plant_img = pygame.transform.scale(plant_img, (50, 50))
    rock_img = pygame.image.load("Images/rock.png").convert_alpha()
    rock_img = pygame.transform.scale(rock_img, (50, 50))
    enemy_images = [rock_img, duck_img, plant_img]

    # Level state
    player = pygame.Rect(100, 275, 50, 50)
    velocity_y  = 0
    GRAVITY = 0.5
    camera_x = 0
    button = pygame.Rect(350, 400, 100, 40)
    hover_start_time = None
    enemy1 = pygame.Rect(500, floor.top - 40, 50, 50)
    enemy1_world_x  = 500
    enemy1_img_index = random.randint(0, len(enemy_images) - 1)
    enemy_count = 0
    game_visible = False
    button_visible = True

    while True:
        # Background
        bg_w   = background.get_width()
        offset = -(camera_x % bg_w)
        screen.blit(background, (offset, 0))
        screen.blit(background, (offset + bg_w, 0))

        g  = gesture["type"]
        dx = direction["x"] * SPEED
        dy = direction["y"] * SPEED

        # Player movement
        if dx > 0:
            if player.right < SCROLL_EDGE - 100:
                player.move_ip(dx, 0)
            else:
                camera_x += SPEED * 4
        elif dx < 0 and player.left > 0:
            player.move_ip(dx, 0)
        if dy < 0 and player.top > 0:
            if player.right < SCROLL_EDGE:
                player.move_ip(3, dy * 3.2)
            else:
                camera_x += SPEED
        elif dy > 0 and player.bottom < SCREEN_H:
            player.move_ip(0, dy)

        # Cursor
        if g == "open":
            cx, cy = cursor_pos["x"], cursor_pos["y"]
            pygame.draw.circle(screen, (0, 200, 255), (cx, cy), 14, 3)
            pygame.draw.line(screen, (0, 200, 255), (cx-20, cy), (cx+20, cy), 2)
            pygame.draw.line(screen, (0, 200, 255), (cx, cy-20), (cx, cy+20), 2)

        # Button
        if button_visible:
            pygame.draw.rect(screen, (0, 0, 0), button, border_radius=8)
            screen.blit(small.render("START!", True, (255,255,255)), (button.x+10, button.y+10))
            screen.blit(font.render("Game Mechanics:", True, (0,0,0)), (button.x-300, button.y-200))
            screen.blit(small.render("Point: up, down, left, right (point in the direction you want!)", True, (0,0,0)), (button.x-220, button.y-150))
            screen.blit(small.render("Hand Open: Cursor mode! (move hand to move cursor)", True, (0,0,0)), (button.x-220, button.y-100))
            screen.blit(large.render("Into The Deep", True, (0,0,0)), (button.x-100, button.y-350))

        # Game logic
        if game_visible:
            enemy1_world_x -= SPEED * random.uniform(0.8, 4)

            if enemy1_world_x < camera_x - 100:
                enemy1_world_x   = camera_x + 900
                enemy1.y         = floor.top - 40
                enemy1_img_index = random.randint(0, len(enemy_images) - 1)
                enemy_count      += 1

            enemy1.x = enemy1_world_x - camera_x

            screen.blit(sprite_img, player.topleft)
            screen.blit(floor_img, floor.topleft)
            screen.blit(enemy_images[enemy1_img_index], enemy1.topleft)

            velocity_y += GRAVITY
            player.y   += int(velocity_y)

            if player.colliderect(floor):
                player.y   = floor.top - player.height
                velocity_y = 0

            if player.colliderect(enemy1):
                game_visible     = False
                button_visible   = True
                player.topleft   = (100, 275)
                enemy1_world_x   = camera_x + 900
                enemy1.y         = floor.top - 40
                enemy_count      = 0  # Reset on death

        # Level complete
        if enemy_count >= 5:
            return True  # Win — caller can load level_2

        # Hover button activation
        if g == "open":
            cx, cy = cursor_pos["x"], cursor_pos["y"]
            if button.collidepoint(cx, cy):
                screen.blit(font.render("hovering", True, (255,255,255)), (10, 50))
                if hover_start_time is None:
                    hover_start_time = pygame.time.get_ticks()
                hover_duration = pygame.time.get_ticks() - hover_start_time
                screen.blit(font.render(str(hover_duration), True, (255,255,255)), (10, 70))

                if hover_duration >= 2000:
                    button_visible   = False
                    game_visible     = True
                    enemy_count      = 0
                    camera_x         = 0
                    player.topleft   = (100, 275)
                    enemy1_world_x   = 500
                    enemy1_img_index = random.randint(0, len(enemy_images) - 1)
                else:
                    progress  = hover_duration / 2000
                    bar_width = int(button.width * progress)
                    pygame.draw.rect(screen, (0, 200, 255), (button.x, button.y + button.height - 5, bar_width, 5))
            else:
                hover_start_time = None

        # HUD labels
        gesture_labels = {
            "point": "POINTING - moving block",
            "open":  "OPEN HAND - cursor mode",
            "fist":  "FIST!",
            "none":  "No hand detected",
        }
        label_color = GESTURE_COLORS.get(g, (150, 150, 150))
        screen.blit(font.render(gesture_labels.get(g, ""), True, label_color), (10, 10))

        if g == "point":
            arrow_map = {(1,0):"→", (-1,0):"←", (0,1):"↓", (0,-1):"↑", (0,0):"-"}
            arrow = arrow_map.get((direction["x"], direction["y"]), "")
            screen.blit(small.render(f"Direction: {arrow}", True, (200,200,200)), (10, 38))

        screen.blit(font.render(f"Enemies dodged: {enemy_count}/5", True, (255,255,255)), (10, SCREEN_H - 30))

        # Camera overlay
        with frame_lock:
            surf = camera_frame["surface"]
        if surf:
            ox = SCREEN_W - OVERLAY_W - PAD
            oy = SCREEN_H - OVERLAY_H - PAD
            pygame.draw.rect(screen, (40, 40, 60),  (ox-2, oy-400, OVERLAY_W+4, OVERLAY_H+4))
            pygame.draw.rect(screen, label_color,   (ox-2, oy-400, OVERLAY_W+4, OVERLAY_H+4), 2)
            screen.blit(surf, (ox, oy-398))
            screen.blit(small.render("HAND CAM", True, (100,100,120)), (ox+4, oy+200))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Quit

        pygame.display.update()
        clock.tick(60)

# =============================================================Level 2==============================================================
def level_2():
    background = pygame.image.load("Images/underground_background.png").convert()
    background = pygame.transform.scale(background, (SCREEN_W, SCREEN_H))

    #fossil images
    skull_img = pygame.image.load("Images/skull.png").convert_alpha()
    skull_img = pygame.transform.scale(skull_img, (100, 100))
    claw_img = pygame.image.load("Images/claw.png").convert_alpha()
    claw_img = pygame.transform.scale(claw_img, (100, 100))
    bone_img = pygame.image.load("Images/bone.png").convert_alpha()
    bone_img = pygame.transform.scale(bone_img, (100, 100))

    # sprite image
    sprite_img = pygame.image.load("Images/fishy.png").convert_alpha()
    sprite_img = pygame.transform.scale(sprite_img, (50, 50))

    player = pygame.Rect(100, 275, 50, 50)
    hover_start_time = None

    claw_visible  = True
    bone_visible  = True
    skull_visible = True

    claw_rect  = pygame.Rect(random.randint(0, SCREEN_W - 200), random.randint(0, SCREEN_H - 200), 100, 100)
    bone_rect  = pygame.Rect(random.randint(0, SCREEN_W - 200), random.randint(0, SCREEN_H - 200), 100, 100)
    skull_rect = pygame.Rect(random.randint(0, SCREEN_W - 200), random.randint(0, SCREEN_H - 200), 100, 100)
    
   
    
    while True:
        screen.blit(background, (0, 0))
        
        #mining logic
        if claw_visible:
            screen.blit(claw_img, claw_rect.topleft)
        if bone_visible:
            screen.blit(bone_img, bone_rect.topleft)
        if skull_visible:
            screen.blit(skull_img, skull_rect.topleft)
        

        screen.blit(sprite_img, player.topleft)

        g = gesture["type"]

        # Movement
        dx = direction["x"] * SPEED
        dy = direction["y"] * SPEED
        if dx < 0 and player.left > 0:               player.move_ip(dx, 0)
        elif dx > 0 and player.right < SCREEN_W:     player.move_ip(dx, 0)
        if dy < 0 and player.top > 0:                player.move_ip(0, dy)
        elif dy > 0 and player.bottom < SCREEN_H:    player.move_ip(0, dy)

        # Cursor
        if g == "open":
            cx, cy = cursor_pos["x"], cursor_pos["y"]
            pygame.draw.circle(screen, (0, 200, 255), (cx, cy), 14, 3)
            pygame.draw.line(screen, (0, 200, 255), (cx-20, cy), (cx+20, cy), 2)
            pygame.draw.line(screen, (0, 200, 255), (cx, cy-20), (cx, cy+20), 2)

        

        #checking if player collides with any fossil
        if player.colliderect(claw_rect):
            if g == "open":
                cx, cy = cursor_pos["x"], cursor_pos["y"]
                if claw_rect.collidepoint(cx, cy):
                    screen.blit(font.render("hovering", True, (255,255,255)), (10, 50))
                    if hover_start_time is None:
                        hover_start_time = pygame.time.get_ticks()
                    hover_duration = pygame.time.get_ticks() - hover_start_time
                    screen.blit(font.render(str(hover_duration), True, (255,255,255)), (10, 70))

                    if hover_duration >= 2000:
                        claw_visible = False  # "Mine" the claw fossil
                    else:
                        progress  = hover_duration / 2000
                        bar_width = int(claw_rect.width * progress)
                        pygame.draw.rect(screen, (0, 200, 255), (claw_rect.x, claw_rect.y + claw_rect.height - 5, bar_width, 5))
                else:
                    hover_start_time = None # Level complete
        if player.colliderect(bone_rect):
            #hover_start_time = None
            if g == "open":
                cx, cy = cursor_pos["x"], cursor_pos["y"]
                if bone_rect.collidepoint(cx, cy):
                    screen.blit(font.render("hovering", True, (255,255,255)), (10, 50))
                    if hover_start_time is None:
                        hover_start_time = pygame.time.get_ticks()
                    hover_duration = pygame.time.get_ticks() - hover_start_time
                    screen.blit(font.render(str(hover_duration), True, (255,255,255)), (10, 70))

                    if hover_duration >= 2000:
                        bone_visible = False  # "Mine" the claw fossil
                        hover_start_time = None # Reset for next fossil
                    else:
                        progress  = hover_duration / 2000
                        bar_width = int(claw_rect.width * progress)
                        pygame.draw.rect(screen, (0, 200, 255), (bone_rect.x, bone_rect.y + bone_rect.height - 5, bar_width, 5))
                else:
                    hover_start_time = None # Level complete 
        if player.colliderect(skull_rect):
            if g == "open":
                cx, cy = cursor_pos["x"], cursor_pos["y"]
                if skull_rect.collidepoint(cx, cy):
                    screen.blit(font.render("hovering", True, (255,255,255)), (10, 50))
                    if hover_start_time is None:
                        hover_start_time = pygame.time.get_ticks()
                    hover_duration = pygame.time.get_ticks() - hover_start_time
                    screen.blit(font.render(str(hover_duration), True, (255,255,255)), (10, 70))

                    if hover_duration >= 2000:
                        skull_visible = False  # "Mine" the claw fossil
                        hover_start_time = None # Reset for next fossil
                    else:
                        progress  = hover_duration / 2000
                        bar_width = int(skull_rect.width * progress)
                        pygame.draw.rect(screen, (0, 200, 255), (skull_rect.x, skull_rect.y + skull_rect.height - 5, bar_width, 5))
                else:
                    hover_start_time = None # Level complete  

        # HUD labels
        gesture_labels = {
            "point": "POINTING - moving block",
            "open":  "OPEN HAND - cursor mode",
            "fist":  "FIST!",
            "none":  "No hand detected",
        }
        label_color = GESTURE_COLORS.get(g, (150, 150, 150))
        screen.blit(font.render(gesture_labels.get(g, ""), True, label_color), (10, 10))

        if g == "point":
            arrow_map = {(1,0):"→", (-1,0):"←", (0,1):"↓", (0,-1):"↑", (0,0):"-"}
            arrow = arrow_map.get((direction["x"], direction["y"]), "")
            screen.blit(small.render(f"Direction: {arrow}", True, (200,200,200)), (10, 38))

        # Camera overlay
        with frame_lock:
            surf = camera_frame["surface"]
        if surf:
            ox = SCREEN_W - OVERLAY_W - PAD
            oy = SCREEN_H - OVERLAY_H - PAD
            pygame.draw.rect(screen, (40, 40, 60),  (ox-2, oy-400, OVERLAY_W+4, OVERLAY_H+4))
            pygame.draw.rect(screen, label_color,   (ox-2, oy-400, OVERLAY_W+4, OVERLAY_H+4), 2)
            screen.blit(surf, (ox, oy-398))
            screen.blit(small.render("HAND CAM", True, (100,100,120)), (ox+4, oy+200))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Quit

        pygame.display.update()
        clock.tick(60)
    
    return True  # Return True to indicate level completion

   

# =============================================================Main Loop==============================================================
running = True
while running:
    result = level_1()
    if not result:
        running = False  # Quit was pressed
    if result:
        result = level_2()
        if not result:
            running = False  # Quit was pressed
    # If result is True, level_2() would go here

pygame.quit()
