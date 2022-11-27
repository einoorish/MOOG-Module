import numpy as np
import json
from matplotlib import pyplot as plt


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def linear_interp(p0, p1, t):
    return (1 - t) * p0 + t * p1


def quadratic_interp(p0, p1, p2, t):
    return linear_interp(
        linear_interp(p0, p1, t),
        linear_interp(p1, p2, t),
        t)


def cubical_interp(p0, p1, p2, p3, t):
    return linear_interp(
        quadratic_interp(p0, p1, p2, t),
        quadratic_interp(p1, p2, p3, t),
        t)

# зчитуємо дані з файлу
points = []

for pointJson in json.load(open("15.json"))["curve"]:
    points.append(Point(pointJson[0], pointJson[1]))

pointsCount = len(points)
matrix_n = 2 * (pointsCount - 1)

# матриця для знаходження контрольних точок
matrix = np.zeros((matrix_n, matrix_n))
right_side_vals = []

# початкова точка
# друга похідна = 0
i = 0
matrix[i][i:i + 2] = [2, -1]
right_side_vals.append(Point(points[i].x, points[i].y))

i += 1
# нижній трикутник - нулі, по діагоналі значення чергуються: або (1,1) або (1,-2,2,-1)
while i < pointsCount - 1:
    matrix[2 * i][2 * i - 1] = 1
    matrix[2 * i][2 * i] = 1

    matrix[2 * i - 1][2 * i - 2] = 1
    matrix[2 * i - 1][2 * i - 1] = -2
    matrix[2 * i - 1][2 * i] = 2
    matrix[2 * i - 1][2 * i + 1] = -1

    # значення справа через 1 множаться на 2 (непарні залишаються нулями)    right_side_val_x[2 * i] = 2 * points[i].x
    right_side_vals.append(Point(0, 0))
    right_side_vals.append(Point(2*points[i].x, 2*points[i].y))
    i += 1

# кінцева точка
# друга похідна = 0
matrix[2 * (pointsCount - 1) - 1][2 * (pointsCount - 1) - 2:2 * (pointsCount - 1)] = [-1, 2]
right_side_vals.append(Point(points[pointsCount - 1].x, points[pointsCount - 1].y))

# отримуємо контрольні точки, вирішивши систему
control_points_x = np.linalg.solve(matrix, list(map(lambda p: p.x, right_side_vals)))
control_points_y = np.linalg.solve(matrix, list(map(lambda p: p.y, right_side_vals)))

bezier = []

for i in range(pointsCount - 1):
    for t in np.linspace(0, 1, 100, False):
        bezier.append(Point(
            cubical_interp(points[i].x, control_points_x[2 * i], control_points_x[2 * i + 1], points[i + 1].x, t)            ,
            cubical_interp(points[i].y, control_points_y[2 * i], control_points_y[2 * i + 1], points[i + 1].y, t)
        ))
fig = plt.figure(1)
ax = plt.axes()

# малюємо сплайн
bezier_p = ax.plot(list(map(lambda p: p.x, bezier)), list(map(lambda p: p.y, bezier)), color='red', linewidth=3)
ax.scatter(list(map(lambda p: p.x, points)), list(map(lambda p: p.y, points)), c='red')

# малюємо контрольні точки
ax.scatter(control_points_x, control_points_y, c='green')

plt.show()
