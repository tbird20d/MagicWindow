#!/usr/bin/python3
#
# project objects - control object projection based on webcam head position
#
import cv2
import math
import numpy as np

mode = "none"
mouse_x = 0
mouse_y = 0

head_x = 0
head_y = 0
webcam_w = 0
webcam_h = 0
render_w = 0
render_h = 0

# mid_x = 0
# mid_y = 0

# Cube size
size = 300
# Cube z
cube_z = 700
cam_pos = [0, 0, -1200]

# Red
front_color = (0, 0, 255)
# Blue
back_color = (255, 0, 0)
# Green
vertex_color = (0, 255, 0)
# Purple
purple = (255, 100, 255)
# White
white = (255, 255, 255)


# Return just x, y of a 3d vertex (wiki)
def project_vertex(cam, v):
    Vx = v[0]
    Vy = v[1]
    Vz = v[2]
    Cx = cam[0]
    Cy = cam[1]
    Cz = cam[2]
    Px = Vx - (Vx - Cx)/(Vz - Cz) * Vz
    Py = Vy - (Vy - Cy)/(Vz - Cz) * Vz

    return int(Px), int(Py)


def project_vertices(vertices, background):
    global cam_pos, render_w, render_h
    screen = background.copy()
    render_h = screen.shape[0]
    render_w = screen.shape[1]

    # Draw Vertices of Cube to plane
    projected_vertices = []
    print("Starting Vertices: ")
    for vertex in vertices:
        v = project_vertex(cam_pos, vertex)
        x = int(v[0] + render_w/2)
        y = int(v[1] + render_h/2)
        cv2.circle(screen, (int(x), int(y)), 2, vertex_color, -1)

        projected_vertices.append((x, y))
    print("Done")

    l_t = 2

    # Draw Lines
    # back square
    cv2.line(screen, projected_vertices[4], projected_vertices[5], back_color, l_t)
    cv2.line(screen, projected_vertices[5], projected_vertices[6], back_color, l_t)
    cv2.line(screen, projected_vertices[6], projected_vertices[7], back_color, l_t)
    cv2.line(screen, projected_vertices[7], projected_vertices[4], back_color, l_t)

    # connecting lines
    cv2.line(screen, projected_vertices[0], projected_vertices[4], purple, l_t)
    cv2.line(screen, projected_vertices[1], projected_vertices[5], purple, l_t)
    cv2.line(screen, projected_vertices[2], projected_vertices[6], purple, l_t)
    cv2.line(screen, projected_vertices[3], projected_vertices[7], purple, l_t)

    # front square
    cv2.line(screen, projected_vertices[0], projected_vertices[1], front_color, l_t)
    cv2.line(screen, projected_vertices[1], projected_vertices[2], front_color, l_t)
    cv2.line(screen, projected_vertices[2], projected_vertices[3], front_color, l_t)
    cv2.line(screen, projected_vertices[3], projected_vertices[0], front_color, l_t)

    cv2.circle(screen, (int(cam_pos[0] + render_w/2), int(cam_pos[1] + render_h/2)), 1, white, -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(screen, "Camera position is " + str(int(cam_pos[0])) + ', ' + str(int(cam_pos[1])),
                (50, render_h - 20), font, 1, white)

    cv2.imshow('Cube', screen)


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


# def draw_cube(cube, background):
#     screen = background.copy()
#
#     # Draw Vertices of Original Cube
#     for vertex in cube:
#         cv2.circle(screen, project_vertex(vertex), 5, vertex_color, -1)
#
#     # Draw Lines
#     # back square
#     cv2.line(screen, project_vertex(cube[4]), project_vertex(cube[5]), back_color, 3)
#     cv2.line(screen, project_vertex(cube[5]), project_vertex(cube[6]), back_color, 3)
#     cv2.line(screen, project_vertex(cube[6]), project_vertex(cube[7]), back_color, 3)
#     cv2.line(screen, project_vertex(cube[7]), project_vertex(cube[4]), back_color, 3)
#
#     # connecting lines
#     cv2.line(screen, project_vertex(cube[0]), project_vertex(cube[4]), purple, 3)
#     cv2.line(screen, project_vertex(cube[1]), project_vertex(cube[5]), purple, 3)
#     cv2.line(screen, project_vertex(cube[2]), project_vertex(cube[6]), purple, 3)
#     cv2.line(screen, project_vertex(cube[3]), project_vertex(cube[7]), purple, 3)
#
#     # front square
#     cv2.line(screen, project_vertex(cube[0]), project_vertex(cube[1]), front_color, 3)
#     cv2.line(screen, project_vertex(cube[1]), project_vertex(cube[2]), front_color, 3)
#     cv2.line(screen, project_vertex(cube[2]), project_vertex(cube[3]), front_color, 3)
#     cv2.line(screen, project_vertex(cube[3]), project_vertex(cube[0]), front_color, 3)
#
#     font = cv2.FONT_HERSHEY_DUPLEX
#     cv2.putText(screen, "Current Mode: " + mode + ".", (50, 50), font, 1, white, 1, cv2.LINE_AA)
#     cv2.putText(screen, str(mouse_x) + ", " + str(mouse_y), (50, 100), font, 1, white, 1, cv2.LINE_AA)
#
#     cv2.imshow('Cube', screen)


def create_cube(size):
    # shift = size / 4
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


def show_webcam(c, fc, ec):
    global head_x, head_y, webcam_w, webcam_h
    ret, frame = c.read()
    webcam_h = frame.shape[0]
    webcam_w = frame.shape[1]

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = fc.detectMultiScale(grey, 1.3, 6)
    for (x, y, w, h) in faces:
        head_x = -(x + w / 2 - webcam_w/2)
        head_y = y + h / 2 - webcam_h/2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roiGrey = grey[y:y + h, x:x + w]
        roiColor = frame[y:y + h, x:x + w]

        eyes = ec.detectMultiScale(roiGrey, 1.10, 6)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roiColor, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # cv2.circle(frame, (int(head_x + webcam_w/2), int(head_y + webcam_h/2)), 4, white, -1)
    # font = cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(frame, "Head position is " + str(head_x) + ', ' + str(head_y), (50, webcam_h - 20), font, 1, white)
    #
    # cv2.imshow('WebCam', frame)


def set_cam_pos():
    global head_x, head_y
    global webcam_w, webcam_h
    global render_w, render_h
    global cam_pos

    # noinspection PyTypeChecker
    cam_pos[0] = head_x * render_w / webcam_w
    # noinspection PyTypeChecker
    cam_pos[1] = head_y * render_h / webcam_h


def main():
    global mode
    global render_w, render_h
    global size, cube_z

    # global mid_x, mid_y

    #screen = cv2.imread('assets/blackScreen.jpg', -1)
    #screen = cv2.resize(screen, (1000, 800))
    screen = np.zeros(shape=[1000, 800, 3], dtype=np.uint8)
    cv2.imshow('Cube', screen)
    render_h = screen.shape[0]
    render_w = screen.shape[1]
    # mid_x = int(render_w / 2)
    # mid_y = int(render_h / 2)

    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    show_webcam(cap, faceCascade, eyeCascade)

    # curX = mid_x
    # curY = mid_y
    # deltaX = 9 / 6
    # deltaY = 1
    scaleFactor = 1.0

    yaw = 0
    pitch = 0
    roll = 0

    cv2.circle(screen, (int(render_w/2), int(render_h/2)), 2, vertex_color, -1)

    oCube = create_cube(size)
    oCube = translate(oCube, 0, 0, cube_z)

    cv2.setMouseCallback('Cube', set_mouse_pos)
    while True:
        mode = get_mode_from_key(1)

        if mode == "translate":
            # curX = mouse_x
            # curY = mouse_y
            curX = head_x
            curY = head_y

        elif mode == "scale":
            scaleFactor = (mouse_x * 3 / render_w) + .3

        elif mode == "rotate":
            yaw = -((mouse_x * math.pi) / render_w - (math.pi/2))
            pitch = (mouse_y * math.pi) / render_h - (math.pi / 2)

        # scaledCube = scale(oCube, scaleFactor)
        # rotatedCube = rotate(scaledCube, yaw, pitch, roll)
        # translatedCube = translate(rotatedCube, curX, curY, 0)
        #

        set_cam_pos()
        project_vertices(oCube, screen)
        show_webcam(cap, faceCascade, eyeCascade)

        if mode == "close_window":
            break

    cv2.destroyAllWindows()


main()
