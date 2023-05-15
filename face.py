import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import argparse
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pygame
import sys
from math import *
from tabulate import tabulate

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def align_image(face, image):
    LEFT_EYE = 263
    RIGHT_EYE = 33
    height, width = image.shape[:2]

    a = (
        face.multi_face_landmarks[0].landmark[LEFT_EYE].x * image.shape[1],
        face.multi_face_landmarks[0].landmark[LEFT_EYE].y * image.shape[0],
    )
    b = (
        face.multi_face_landmarks[0].landmark[RIGHT_EYE].x * image.shape[1],
        face.multi_face_landmarks[0].landmark[RIGHT_EYE].y * image.shape[0],
    )

    dx = a[0] - b[0]
    dy = a[1] - b[1]
    angle = atan2(dy, dx) * 180 / pi
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    aligned_image = cv2.warpAffine(image, M, (width, height))

    return aligned_image


def get_pitch(face):
    FOREHEAD = 9
    CHIN = 199

    a = (
        face[FOREHEAD][1],
        face[FOREHEAD][2],
    )
    b = (
        face[CHIN][1],
        face[CHIN][2],
    )
    dy = b[0] - a[0]
    dz = b[1] - a[1]

    return atan2(dz, dy)


def get_roll(face):
    LEFT_EYE = 398
    RIGHT_EYE = 33

    a = (
        face[LEFT_EYE][0],
        face[LEFT_EYE][1],
    )
    b = (
        face[RIGHT_EYE][0],
        face[RIGHT_EYE][1],
    )

    dx = a[0] - b[0]
    dy = a[1] - b[1]

    return atan2(dx, dy) - pi / 2


def get_yaw(face):
    LEFT_EYE = 362
    RIGHT_EYE = 133

    a = (
        face[LEFT_EYE][0],
        face[LEFT_EYE][2],
    )
    b = (
        face[RIGHT_EYE][0],
        face[RIGHT_EYE][2],
    )

    dx = a[0] - b[0]
    dz = a[1] - b[1]

    return atan2(dz, dx)


def average_faces(dir="./output", out="./data"):
    faces = []

    for filename in os.listdir(dir):
        image = cv2.imread(os.path.join(dir, filename))
        face = normalize(image)
        faces.append(face)
    average = np.mean(faces, axis=0, dtype=int)
    average = scale(average)

    with open(os.path.join(out, "average.pickle"), "wb") as f:
        pickle.dump(average, f, pickle.HIGHEST_PROTOCOL)

    return average


def crop_face(image):
    face = get_mesh(image)
    np_face = np.array(get_coords(face.multi_face_landmarks[0], image))
    maxes = np.max(np_face, axis=0)
    mins = np.min(np_face, axis=0)
    return image[mins[1] : maxes[1], mins[0] : maxes[0]]


def draw_mesh(image):
    face = get_mesh(image)
    annotated_image = image.copy()
    for face_landmarks in face.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
    cv2.namedWindow("Face Mesh", cv2.WINDOW_NORMAL)
    cv2.imshow("Face Mesh", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def export_meshes(dir="./input", out="./data/meshes"):
    for filename in os.listdir(dir):
        image = cv2.imread(os.path.join(dir, filename))
        face = normalize(image)
        with open(os.path.join(out, filename.split(".")[0] + ".pickle"), "wb") as f:
            pickle.dump(face, f, pickle.HIGHEST_PROTOCOL)


def extract_faces(dir="./input", out="./output"):
    for filename in os.listdir(dir):
        image = cv2.imread(os.path.join(dir, filename))
        face = get_mesh(image)
        results = align_image(face, image)
        results = crop_face(results)
        cv2.imwrite(os.path.join(out, filename), results)


def face_similarity(a, b):
    dist = np.linalg.norm(a - b, axis=1)
    return np.average(dist)


def get_coords(landmarks, image):
    h, w = image.shape[:2]
    xyz = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    return np.multiply(xyz, [w, h, w]).astype(int)


def get_mesh(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        return face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def multiply_m(a, b):
    return np.dot(a, b).astype(int)


def normalize(image):
    face = get_mesh(image)
    face_points = get_coords(face.multi_face_landmarks[0], image)

    angle_x = angle_y = angle_z = 0
    angle_x = get_pitch(face_points)
    angle_y = get_yaw(face_points)
    angle_z = get_roll(face_points)

    rotation_x = rotation_matrix_x(angle_x)
    rotation_y = rotation_matrix_y(angle_y)
    rotation_z = rotation_matrix_z(angle_z)

    for n, point in enumerate(face_points):
        rotations = multiply_m(
            rotation_z, (multiply_m(rotation_y, multiply_m(rotation_x, point)))
        )
        face_points[n] = rotations

    face_points = scale(face_points)

    return face_points


def rotation_matrix_x(angle):
    return [
        [1, 0, 0],
        [0, cos(angle), -sin(angle)],
        [0, sin(angle), cos(angle)],
    ]


def rotation_matrix_y(angle):
    return [
        [cos(angle), 0, sin(angle)],
        [0, 1, 0],
        [-sin(angle), 0, cos(angle)],
    ]


def rotation_matrix_z(angle):
    return [
        [cos(angle), -sin(angle), 0],
        [sin(angle), cos(angle), 0],
        [0, 0, 1],
    ]


def scale(mesh, size=1000):
    face = np.array(mesh)
    maxes = np.max(face, axis=0)
    mins = np.min(face, axis=0)
    x_range = maxes[0] - mins[0]
    y_range = maxes[1] - mins[1]
    z_range = maxes[2] - mins[2]
    max_range = max(x_range, y_range, z_range)
    x_padding = (size - (x_range * size / max_range)) / 2

    for p in mesh:
        p[0] = (p[0] - mins[0]) * size / max_range + x_padding
        p[1] = (p[1] - mins[1]) * size / max_range
        p[2] = (p[2] - mins[2]) * size / max_range

    return mesh


def visualize_mesh(mesh):
    WINDOW_SIZE = 1000
    ROTATE_SPEED = 0.02
    PROJECTION_MATRIX = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]

    window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()
    clock.tick(60)

    angle_x = angle_y = angle_z = 0

    while True:
        window.fill((0, 0, 0))
        rotation_x = [
            [1, 0, 0],
            [0, cos(angle_x), -sin(angle_x)],
            [0, sin(angle_x), cos(angle_x)],
        ]

        rotation_y = [
            [cos(angle_y), 0, sin(angle_y)],
            [0, 1, 0],
            [-sin(angle_y), 0, cos(angle_y)],
        ]

        rotation_z = [
            [cos(angle_z), -sin(angle_z), 0],
            [sin(angle_z), cos(angle_z), 0],
            [0, 0, 1],
        ]

        points = [0 for _ in range(len(mesh))]
        i = 0
        for point in mesh:
            rotate_x = multiply_m(rotation_x, point)
            rotate_y = multiply_m(rotation_y, rotate_x)
            rotate_z = multiply_m(rotation_z, rotate_y)
            point_2d = multiply_m(PROJECTION_MATRIX, rotate_z)

            x = point_2d[0]
            y = point_2d[1]

            points[i] = (x, y)
            i += 1
            pygame.draw.circle(window, (255, 255, 255), (x, y), 1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit(0)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                angle_y = angle_x = angle_z = 0
            if keys[pygame.K_a]:
                angle_y += ROTATE_SPEED
            if keys[pygame.K_d]:
                angle_y -= ROTATE_SPEED
            if keys[pygame.K_w]:
                angle_x += ROTATE_SPEED
            if keys[pygame.K_s]:
                angle_x -= ROTATE_SPEED
            if keys[pygame.K_q]:
                angle_z -= ROTATE_SPEED
            if keys[pygame.K_e]:
                angle_z += ROTATE_SPEED

        pygame.display.update()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--average", action="store_true", help="average face meshes"
    )
    parser.add_argument(
        "-c", "--compare", action="store_true", help="compare faces to averaged face"
    )
    parser.add_argument(
        "-d", "--draw", action="store_true", help="draw face mesh over image"
    )
    parser.add_argument(
        "-e", "--export", action="store_true", help="export face meshes"
    )
    parser.add_argument(
        "-v", "--viz", action="store_true", help="visualize averaged face mesh"
    )
    parser.add_argument("-x", "--extract", action="store_true", help="extract faces")
    args = parser.parse_args()
    return args


def main():
    inputs = parse_args()

    if inputs.draw:
        for filename in os.listdir("./input"):
            image = cv2.imread(os.path.join("./input", filename))
            draw_mesh(image)

    if inputs.extract:
        extract_faces()

    if inputs.export:
        export_meshes()

    if inputs.average:
        average = average_faces(dir="./input")

    if inputs.compare:
        with open("./data/average.pickle", "rb") as f:
            average = pickle.load(f)

        table = [["File name", "Euclidean Distance"]]

        for filename in os.listdir("./compare"):
            image = cv2.imread(os.path.join("./compare", filename))
            table.append([filename, face_similarity(average, normalize(image))])
        print(tabulate(table, headers="firstrow", tablefmt="orgtbl", floatfmt=".2f"))

    if inputs.viz:
        for filename in os.listdir("./compare"):
            with open("./data/average.pickle", "rb") as f:
                average = pickle.load(f)
            visualize_mesh(average)


if __name__ == "__main__":
    main()
