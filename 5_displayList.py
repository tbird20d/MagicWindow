#!/usr/bin/python3
#
# Show a 3D scene using a display list, and adjust
# the camera position using the detected head position of
# the user.

import cv2
import numpy as np
import math

mode = "none"
mouse_x = 0
mouse_y = 0

head_x = 0
head_y = 0
head_z = -1200
webcam_w = 0
webcam_h = 0
render_w = 0
render_h = 0

# Cube size
size = 300
# Cube z
cube_z = 700
cam_pos = [0, 0, 0]


red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
purple = (255, 0, 135)
white = (255, 255, 255)
brown = (0, 57, 119)


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


def scale(cube, factor):
    new_cube = []
    for x, y, z in cube:
        new_cube.append((int(x * factor), int(y * factor), int(z * factor)))
    return new_cube


def translate(dlist, fx, fy, fz):
    new_dlist = []
    for item in dlist:
        if item[0] == 'p':
            new_item = ('p', (item[1][0] + fx, item[1][1] + fy, item[1][2] + fz))
            new_dlist.append(new_item)
        else:
            new_dlist.append(item)
    return new_dlist


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


def set_mouse_pos(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
#
#
#
#


# Return just x, y of a 3d vertex (wiki)
def project_vertex(cam, v):
    global render_w, render_h
    Vx = v[0]
    Vy = v[1]
    Vz = v[2]
    Cx = cam[0]
    Cy = cam[1]
    Cz = cam[2]
    Px = Vx - (Vx - Cx)/(Vz - Cz) * Vz
    Py = Vy - (Vy - Cy)/(Vz - Cz) * Vz

    return int(Px + render_w/2), int(Py + render_h/2)


def project_list(dlist):
    global cam_pos, render_w, render_h

    plist = []
    for item in dlist:
        if item[0] == 'p':
            new_item = ('p', project_vertex(cam_pos, item[1]))
            plist.append(new_item)

        else:
            plist.append(item)

    return plist


def render(background, plist):
    global cam_pos, render_w, render_h
    screen = background.copy()

    for item in plist:
        if item[0] == 'p':
            # cv2.circle(screen, item[1], 2, green, -1)
            pass

        elif item[0] == 'l':
            p1_ref = item[1][0]
            p2_ref = item[1][1]
            color = item[1][2]
            p1 = plist[p1_ref][1]
            p2 = plist[p2_ref][1]

            cv2.line(screen, p1, p2, color, 2)

        elif item[0] == 't':
            p1_ref = item[1][0]
            p2_ref = item[1][1]
            p3_ref = item[1][2]
            color = item[1][3]
            p1 = plist[p1_ref][1]
            p2 = plist[p2_ref][1]
            p3 = plist[p3_ref][1]

            triangle_cnt = np.array([p1, p2, p3])
            cv2.drawContours(screen, [triangle_cnt], 0, color, -1)

        elif item[0] == 'r':
            p1_ref = item[1][0]
            p2_ref = item[1][1]
            color = item[1][2]
            p1 = plist[p1_ref][1]
            p2 = plist[p2_ref][1]

            cv2.rectangle(screen, p1, p2, color, -1)

    cv2.circle(screen, (int(cam_pos[0] + render_w / 2), int(cam_pos[1] + render_h / 2)), 1, white, -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(screen,
                "Camera position is " + str(int(cam_pos[0])) + ', ' + str(int(cam_pos[1])) + ', ' + str(int(cam_pos[2])),
                (50, render_h - 20), font, 1, white)
    cv2.imshow('Cube', screen)


def dlist_append(dlist1, dlist2):
    new_dlist = dlist1
    index_offset = len(new_dlist)
    for item in dlist2:
        if item[0] == 'l':
            new_item = ('l', (item[1][0] + index_offset, item[1][1] + index_offset, item[1][2]))
            new_dlist.append(new_item)
        elif item[0] == 't':
            new_item = ('t', (item[1][0] + index_offset, item[1][1] + index_offset,
                              item[1][2] + index_offset, item[1][3]))
            new_dlist.append(new_item)
        elif item[0] == 'r':
            new_item = ('r', (item[1][0] + index_offset, item[1][1] + index_offset, item[1][2]))
            new_dlist.append(new_item)
        else:
            new_dlist.append(item)

    return new_dlist


def create_cube(size):
    # x, y, z  aka  3d vertex
    oPoint0 = -size / 2, -size / 2, -size / 2
    oPoint1 = size / 2, -size / 2, -size / 2
    oPoint2 = size / 2, size / 2, -size / 2
    oPoint3 = -size / 2, size / 2, -size / 2

    oPoint4 = oPoint0[0], oPoint0[1], size / 2
    oPoint5 = oPoint1[0], oPoint1[1], size / 2
    oPoint6 = oPoint2[0], oPoint2[1], size / 2
    oPoint7 = oPoint3[0], oPoint3[1], size / 2

    dlist = [
        ('p', oPoint0),
        ('p', oPoint1),
        ('p', oPoint2),
        ('p', oPoint3),
        ('p', oPoint4),
        ('p', oPoint5),
        ('p', oPoint6),
        ('p', oPoint7),
        ('l', (4, 5, blue)),
        ('l', (5, 6, blue)),
        ('l', (6, 7, blue)),
        ('l', (7, 4, blue)),
        ('l', (0, 4, purple)),
        ('l', (1, 5, purple)),
        ('l', (2, 6, purple)),
        ('l', (3, 7, purple)),
        ('l', (0, 1, red)),
        ('l', (1, 2, red)),
        ('l', (2, 3, red)),
        ('l', (3, 0, red)),
    ]
    return dlist


def create_r_pyramid(x, y, z, bw, h):
    dlist = [
        ('p', (x - bw / 2, y, z + bw / 2)),
        ('p', (x + bw / 2, y, z + bw / 2)),
        ('p', (x + bw / 2, y, z - bw / 2)),
        ('p', (x - bw / 2, y, z - bw / 2)),
        ('p', (x, y-h, z)),
        ('t', (0, 1, 4, purple)),
        ('t', (1, 2, 4, purple)),
        ('t', (2, 3, 4, purple)),
        ('t', (3, 1, 4, purple)),
        # outline
        ('l', (0, 1, white)),
        ('l', (1, 2, white)),
        ('l', (2, 3, white)),
        ('l', (3, 0, white)),
        ('l', (0, 4, white)),
        ('l', (1, 4, white)),
        ('l', (2, 4, white)),
        ('l', (3, 4, white))
    ]
    return dlist


def create_tree(x, y, z, tw, th):
    trunk_h = th/5
    trunk_w = tw/5
    dlist = [
        ('p', (x-trunk_w/2, y, z)),
        ('p', (x+trunk_w/2, y-trunk_h, z)),
        ('r', (0, 1, brown)),

        ('p', (x-tw/2, y-trunk_h, z)),
        ('p', (x+tw/2, y-trunk_h, z)),
        ('p', (x, y-trunk_h-th, z)),
        ('t', (3, 4, 5, green))
    ]
    return dlist


def create_horizon():
    oPoint0 = -20000, 0, 10000
    oPoint1 = 20000, 0, 10000

    dlist = [
        ('p', oPoint0),
        ('p', oPoint1),
        ('l', (0, 1, brown)),
    ]

    return dlist


def show_webcam(c, fc, ec):
    global head_x, head_y, head_z
    global webcam_w, webcam_h
    ret, frame = c.read()
    webcam_h = frame.shape[0]
    webcam_w = frame.shape[1]

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = fc.detectMultiScale(grey, 1.3, 6)
    for (x, y, w, h) in faces:
        head_x = -(x + w / 2 - webcam_w/2)
        head_y = y + h / 2 - webcam_h/2
        # head_z = -(3400 - w * 50/3)
        # head_z = -1200 - 4 * (math.pow(2, (w-520)/-50))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roiGrey = grey[y:y + h, x:x + w]
        roiColor = frame[y:y + h, x:x + w]

        # eyes = ec.detectMultiScale(roiGrey, 1.10, 6)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roiColor, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # cv2.circle(frame, (int(head_x + webcam_w/2), int(head_y + webcam_h/2)), 4, white, -1)
    # font = cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(frame, "Head position is " + str(head_x) + ', ' + str(head_y) + str(),
    #             (50, webcam_h - 20), font, 1, white)
    # cv2.imshow('WebCam', frame)


def set_cam_pos():
    global head_x, head_y, head_z
    global webcam_w, webcam_h
    global render_w, render_h
    global cam_pos

    # noinspection PyTypeChecker
    cam_pos[0] = head_x * render_w / webcam_w
    # noinspection PyTypeChecker
    cam_pos[1] = head_y * render_h / webcam_h
    cam_pos[2] = head_z


def main():
    global mode
    global render_w, render_h
    global size, cube_z

    #screen = cv2.imread('assets/blackScreen.jpg', -1)
    #screen = cv2.resize(screen, (1500, 860))
    screen = np.zeros(shape=[860, 1500, 3], dtype=np.uint8)
    cv2.imshow('Cube', screen)
    render_h = screen.shape[0]
    render_w = screen.shape[1]

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

    # Center of Screen in green
    cv2.circle(screen, (int(render_w/2), int(render_h/2)), 2, green, -1)

    dlist = create_horizon()
    dlist_append(dlist, [
        ('p', (-500, 0, 10000)),
        ('p', (-500, 0, 0)),
        ('l', (0, 1, brown)),
        ('p', (500, 0, 10000)),
        ('p', (500, 0, 0)),
        ('l', (3, 4, brown))
    ])

    for i in range(5):
        x = 500 + i*300
        z = 3500 - i * 500
        tree = create_tree(x, 0, z, 150, 225)
        dlist_append(dlist, tree)

    for i in range(5):
        x = -500 - i * 300
        z = 5000 - i * 700
        tree = create_tree(x, 0, z, 150, 225)
        dlist_append(dlist, tree)

    cube3 = create_cube(int(size))
    cube3 = translate(cube3, -size - 50, -size/2, cube_z + 1200)
    dlist_append(dlist, cube3)

    cube2 = create_cube(size)
    cube2 = translate(cube2, size + 50, -size/2, cube_z + 800)
    dlist_append(dlist, cube2)

    cube1 = create_cube(size)
    cube1 = translate(cube1, 0, -size/2, cube_z)
    dlist_append(dlist, cube1)

    pyramid1 = create_r_pyramid(-300, 0, 500, 100, 200)
    dlist_append(dlist, pyramid1)

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
        plist = project_list(dlist)
        render(screen, plist)
        show_webcam(cap, faceCascade, eyeCascade)

        if mode == "close_window":
            break

    cv2.destroyAllWindows()


main()
