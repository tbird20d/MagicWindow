#!/usr/bin/python3
#
import cv2
import math
import numpy as np

mode = "none"
mouse_x = 0
mouse_y = 0

# """ Return just x, y of a 3d vertex """
def project(v):
    return int(v[0]), int(v[1])


def scale(cube, factor):
    new_cube = []
    for x, y, z in cube:
        new_cube.append((int(x * factor), int(y * factor), int(z * factor)))
    return new_cube


def translate(cube, fx, fy, fz):
    new_cube = []
    for x, y, z in cube:
        new_cube.append((x + fx, y + fy, z + fz))
    return new_cube


def rotate(cube, alpha, beta, gamma):
    cosA = math.cos(alpha)
    sinA = math.sin(alpha)
    cosB = math.cos(beta)
    sinB = math.sin(beta)
    cosG = math.cos(gamma)
    sinG = math.sin(gamma)

    new_cube = []

    for x, y, z in cube:
        newX = x * (cosG * cosA) + y * (cosG * sinA * sinB - sinG * cosB) + z * (cosG * sinA * cosB + sinG * sinB)
        newY = x * (sinG * cosA) + y * (sinG * sinA * sinB + cosG * cosB) + z * (sinG * sinA * cosB - cosG * sinB)
        newZ = x * (-sinA) + y * (cosA * sinB) + z * (cosA * cosB)

        new_cube.append((newX, newY, newZ))

    return new_cube


def draw_cube(cube, background):
    screen = background.copy()

    # """ Red """
    front_color = (0, 0, 255)
    # """ Blue """
    back_color = (255, 0, 0)
    # """ Green """
    vertex_color = (0, 255, 0)
    # """ Purple """
    purple = (255, 100, 255)
    # """ White """
    white = (255, 255, 255)

    # """ Draw Vertices of Original Cube """
    for vertex in cube:
        cv2.circle(screen, project(vertex), 5, vertex_color, -1)

    # """ Draw Lines """
    # """ back square """
    cv2.line(screen, project(cube[4]), project(cube[5]), back_color, 3)
    cv2.line(screen, project(cube[5]), project(cube[6]), back_color, 3)
    cv2.line(screen, project(cube[6]), project(cube[7]), back_color, 3)
    cv2.line(screen, project(cube[7]), project(cube[4]), back_color, 3)

    # """ connecting lines """
    cv2.line(screen, project(cube[0]), project(cube[4]), purple, 3)
    cv2.line(screen, project(cube[1]), project(cube[5]), purple, 3)
    cv2.line(screen, project(cube[2]), project(cube[6]), purple, 3)
    cv2.line(screen, project(cube[3]), project(cube[7]), purple, 3)

    # """ front square """
    cv2.line(screen, project(cube[0]), project(cube[1]), front_color, 3)
    cv2.line(screen, project(cube[1]), project(cube[2]), front_color, 3)
    cv2.line(screen, project(cube[2]), project(cube[3]), front_color, 3)
    cv2.line(screen, project(cube[3]), project(cube[0]), front_color, 3)

    font = cv2.FONT_HERSHEY_DUPLEX
    img = cv2.putText(screen, "Press t,s,r for mode, and <esc> to exit", (50, 10), font, 1, white, 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_DUPLEX
    img = cv2.putText(screen, "Current Mode: " + mode + ".", (50, 50), font, 1, white, 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_DUPLEX
    img = cv2.putText(screen, str(mouse_x) + ", " + str(mouse_y), (50, 100), font, 1, white, 1, cv2.LINE_AA)

    cv2.imshow('Cube', screen)


def create_o_cube(size):
    shift = size / 4
    shift = 0

    # x, y, z  aka  3d vertex
    oPoint1 = -size / 2, -size / 2, -size / 2
    oPoint2 = size / 2, -size / 2, -size / 2
    oPoint3 = size / 2, size / 2, -size / 2
    oPoint4 = -size / 2, size / 2, -size / 2
    fSquare = [oPoint1, oPoint2, oPoint3, oPoint4]

    oPoint5 = oPoint1[0] - shift, oPoint1[1] - shift, size / 2
    oPoint6 = oPoint2[0] - shift, oPoint2[1] - shift, size / 2
    oPoint7 = oPoint3[0] - shift, oPoint3[1] - shift, size / 2
    oPoint8 = oPoint4[0] - shift, oPoint4[1] - shift, size / 2
    bSquare = [oPoint5, oPoint6, oPoint7, oPoint8]

    oCube = fSquare + bSquare
    return oCube


def get_mode_from_key(delay):
    global mode

    k = cv2.waitKey(delay)
    if k & 0xFF == ord('t'):
        mode = "translate"
    if k & 0xFF == ord('s'):
        mode = "scale"
    if k & 0xFF == ord('r'):
        mode = "rotate"
    if k & 0xFF == 27:
        mode = "close_window"

    return mode


def set_mouse_pos(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y


def main():
    global mode

    #screen = cv2.imread('assets/blackScreen.jpg', -1)
    #screen = cv2.resize(screen, (900, 600))
    screen = np.zeros(shape=[900, 600, 3], dtype=np.uint8)
    cv2.imshow('Cube', screen)
    h = screen.shape[0]
    w = screen.shape[1]
    esc = 27

    curX = int(w / 2)
    curY = int(h / 2)
    deltaX = 9 / 6
    deltaY = 1
    scaleFactor = 1.0

    yaw = 0
    pitch = 0
    roll = 0

    size = 100
    oCube = create_o_cube(size)

    # draw_cube(oCube, screen)

    cv2.setMouseCallback('Cube', set_mouse_pos)
    while True:
        mode = get_mode_from_key(1)

        # """Translate Mode"""
        if mode == "translate":
            curX = mouse_x
            curY = mouse_y

        # """ Scale Mode """
        elif mode == "scale":
            scaleFactor = (mouse_x * 3 / w) + .3

        # Rotate Mode
        elif mode == "rotate":
            yaw = -((mouse_x * math.pi) / w - (math.pi/2))
            pitch = (mouse_y * math.pi) / h - (math.pi / 2)

        scaledCube = scale(oCube, scaleFactor)
        rotatedCube = rotate(scaledCube, yaw, pitch, roll)
        translatedCube = translate(rotatedCube, curX, curY, 0)

        draw_cube(translatedCube, screen)

        if mode == "close_window":
            break

    cv2.destroyAllWindows()


main()
