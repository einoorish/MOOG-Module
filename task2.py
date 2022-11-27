import numpy as np
import json
import math
import open3d


def calc_distance_between_points(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


class Surface(object):
    def __init__(self, points, indices, gridSize):
        self.points = points
        self.degree = 3

        self.input_surface_matrix = np.zeros((gridSize[0], gridSize[1], 3))
        for i, q in enumerate(indices):
            self.input_surface_matrix[q[0], q[1]] = points[i]
        self.n = len(self.input_surface_matrix)
        self.k = self.degree

        self.s_column = np.zeros(self.n, dtype=np.float32)
        self.s_row = np.zeros(self.n, dtype=np.float32)

        self.knots_column = np.zeros(self.n + self.k + 1, dtype=np.float32)
        self.knots_row = np.zeros(self.n + self.k + 1, dtype=np.float32)

        self.control_points = np.zeros((self.n, self.n, 3))

        self.nurbs_surface_points = []

        self._calc_s()
        self._generate_knots()
        self._generate_control_points()

    def _calc_s(self):
        d_columns = np.zeros((self.n, self.n), dtype=np.float32)
        d_rows = np.zeros((self.n, self.n), dtype=np.float32)

        for i in range(self.n):
            d_columns[i] = self._calc_distances(self.input_surface_matrix[:, i])
            d_rows[i] = self._calc_distances(self.input_surface_matrix[i])

        for i in range(self.n):
            self.s_column[i] = sum(d_columns[:, i][j] for j in range(self.n)) / self.n
            self.s_row[i] = sum(d_rows[:, i][j] for j in range(self.n)) / self.n

    def _calc_distances(self, points):
        result = np.zeros(len(points), dtype=np.float32)
        result[len(points) - 1] = 1.

        total = sum(calc_distance_between_points(points[i], points[i - 1]) for i in range(1, len(points)))
        for i in range(1, len(points) - 1):
            result[i] = result[i - 1] + calc_distance_between_points(points[i], points[i - 1]) / total
        return result

    def _generate_knots(self):
        for i in range(1, self.n - self.k):
            self.knots_column[i + self.k] = (1. / self.k) * sum(self.s_column[j] for j in range(i, i + self.k))
            self.knots_row[i + self.k] = (1. / self.k) * sum(self.s_row[j] for j in range(i, i + self.k))

        for i in range(self.k + 1):
            self.knots_column[i], self.knots_row[i] = 0., 0.
            self.knots_column[self.n + self.k - i], self.knots_row[self.n + self.k - i] = 1., 1.

    def _generate_control_points(self):
        q = np.zeros((self.n, self.n, 3))
        basis_function_matrix = np.zeros((self.n, self.n), dtype=np.float32)

        for i in range(self.n):
            for j in range(self.n):
                basis_function_matrix[i, j] = self.basis_function(self.s_column[i], j, self.k, self.knots_column)
        for i in range(self.n):
            q[:, i] = np.linalg.solve(basis_function_matrix, self.input_surface_matrix[:, i])

        for i in range(self.n):
            for j in range(self.n):
                basis_function_matrix[i, j] = self.basis_function(self.s_row[i], j, self.k, self.knots_row)
        for i in range(self.n):
            self.control_points[i] = np.linalg.solve(basis_function_matrix, q[i])

    def basis_function(self, t, i, k, knot):
        if k == 0:
            if t == knot[i] or knot[i] < t < knot[i + 1]:
                return 1
            elif t == knot[len(knot) - 1] and i == len(knot) - self.degree - 2:
                return 1

        result = 0.

        diff = (knot[i + k] - knot[i])
        if diff != 0:
            result += (t - knot[i]) * self.basis_function(t, i, k - 1, knot) / diff
        diff1 = (knot[i + k + 1] - knot[i + 1])
        if diff1 != 0:
            result += (knot[i + k + 1] - t) * self.basis_function(t, i + 1, k - 1, knot) / diff1

        return result

    def map_to_surface(self, u, v):
        result = np.zeros(3)
        for i in range(self.n):
            curr_sum = 0
            for j in range(self.n):
                curr_sum += self.basis_function(v, j, self.k, self.knots_row) * self.control_points[i, j]
            result += curr_sum * self.basis_function(u, i, self.k, self.knots_column)
        return result

    def make_triangle_mesh(self, p_num):
        u = np.linspace(0, 1, p_num)
        v = np.linspace(0, 1, p_num)
        for i in range(len(u)):
            for j in range(len(v)):
                self.nurbs_surface_points.append(self.map_to_surface(u[i], v[j]))

        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(self.nurbs_surface_points)
        mesh.paint_uniform_color(np.array([211, 211, 211])/255)

        triangles = []
        for i in range(p_num - 1):
            for j in range(p_num - 1):
                triangles.append(np.array([
                    i * p_num + j,
                    i * p_num + j + p_num,
                    i * p_num + j + 1]
                ).astype(np.int32))
                triangles.append(np.array([
                    i * p_num + j + 1,
                    i * p_num + j + p_num,
                    i * p_num + j + p_num + 1]
                ).astype(np.int32))

        mesh.triangles = open3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()

        return mesh


def visualize(surf, points, points_multiplier):
    surface_mesh = surf.make_triangle_mesh(surf.n * points_multiplier)

    p_cloud = open3d.geometry.PointCloud()
    p_cloud.points = open3d.utility.Vector3dVector(points)
    mat = open3d.visualization.rendering.MaterialRecord()
    mat.base_color = [1.0, 0.5, 0.0, 1.0]
    mat.point_size = 9.0
    open3d.visualization.draw([surface_mesh, {'name': 'pcd', 'geometry': p_cloud, 'material': mat}])
    open3d.visualization.draw_geometries([surface_mesh], mesh_show_back_face=True, mesh_show_wireframe=True)


surface_data = json.load(open("15.json"))["surface"]
points, indices, gridSize = surface_data["points"], surface_data["indices"], surface_data["gridSize"]

surface = Surface(points, indices, gridSize)
multiplier = 3
visualize(surface, points, multiplier)
