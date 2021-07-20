import queue

import numpy as np
import cv2
from tqdm import trange
from time import time


class Cluster:
    def __init__(self, no, x, y, l=0, a=0, b=0):
        self.no = no
        self.update(x, y, l, a, b)
        self.pixels = []

    def update(self, x, y, l, a, b):
        self.x = int(x)
        self.y = int(y)
        self.l = l
        self.a = a
        self.b = b

    def get_pixel(self):
        return np.array([self.l, self.a, self.b])

    def __str__(self):
        return "{} {} {} {} {} {}".format(self.no, self.x, self.y, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLIC:
    def __init__(self, image, k, m=10, iter=4):
        self.k = k  # number of superpixels
        self.m = m
        self.iter = iter
        self.image = image
        self.image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
        self.image_height, self.image_width = self.image.shape[:2]
        self.N = self.image_height * self.image_width  # number of pixels
        self.S = int((self.N / self.k) ** 0.5)  # grid_interval
        print(self.N, self.k, self.S)
        print(self.image_height, self.image_width)
        self.clusters = []
        self.cluster_matrix = -1 * np.ones(self.image.shape[:2], dtype=int)
        self.index_mesh = np.mgrid[0:self.image_height, 0:self.image_width].transpose(1, 2, 0)
        self.cluster_image = None

    def __get_gradient(self, x, y):
        current = self.image_lab[y, x, 0]
        next_row = self.image_lab[y + 1, x, 0]
        next_col = self.image_lab[y, x + 1, 0]
        col_grad = ((next_col - current) ** 2) ** 0.5
        rol_grad = ((next_row - current) ** 2) ** 0.5
        return rol_grad + col_grad

    def __find_lowest_gradient_position(self, x, y):
        min_grad = 100
        # cluster = None
        cluster = Cluster(len(self.clusters), x, y, self.image_lab[y, x, 0], self.image_lab[y, x, 1],
                          self.image_lab[y, x, 2])
        for _dx in range(-1, 2):
            for _dy in range(-1, 2):
                dx = x + _dx
                dy = y + _dy
                grad = self.__get_gradient(dx, dy)
                if grad < min_grad:
                    min_grad = grad
                    cluster.update(dy, dx,
                                   self.image_lab[dy, dx, 0],
                                   self.image_lab[dy, dx, 1],
                                   self.image_lab[dy, dx, 2])
        return cluster

    def __init_clusters(self):
        x = y = self.S // 2
        while y < self.image_height:
            while x < self.image_width:
                cluster = self.__find_lowest_gradient_position(x, y)
                self.clusters.append(cluster)
                x += self.S
            x = self.S // 2
            y += self.S

    def clustering(self):
        min_distance = np.full((self.image_height, self.image_width), np.inf)
        for cluster in self.clusters:
            cluster_point = np.array([cluster.l, cluster.a, cluster.b])
            x_min = max(0, cluster.x - 2 * self.S)
            y_min = max(0, cluster.y - 2 * self.S)
            x_max = min(self.image_width, cluster.x + 2 * self.S)
            y_max = min(self.image_height, cluster.y + 2 * self.S)

            x_min, y_min, x_max, y_max = [int(val) for val in [x_min, y_min, x_max, y_max]]

            y_pixels, x_pixels = np.ogrid[y_min:y_max, x_min:x_max]

            crop = self.image[y_min:y_max, x_min:x_max]
            color_distance = np.sqrt(np.sum((crop - cluster_point) ** 2, axis=-1))
            spatial_distance = np.sqrt((y_pixels - cluster.y) ** 2 + (x_pixels - cluster.x) ** 2)
            total_distance = color_distance + (self.m / self.S) * spatial_distance
            min_distance_area = min_distance[y_min:y_max, x_min:x_max]
            idx = total_distance < min_distance_area
            min_distance_area[idx] = total_distance[idx]
            self.cluster_matrix[y_min:y_max, x_min:x_max][idx] = cluster.no

        for cluster in self.clusters:
            idx = self.cluster_matrix == cluster.no
            if len(self.index_mesh[idx]) == 0:
                continue
            cluster.pixels = self.index_mesh[idx].tolist()
            mean_l, mean_a, mean_b = np.mean(self.image_lab[idx], axis=0)
            mean_y, mean_x = np.mean(self.index_mesh[idx], axis=0)
            cluster.update(mean_x, mean_y, mean_l, mean_a, mean_b)

    def find_new_biggest_cluster(self, x, y, cluster_no):
        biggest_area = 0
        new_cluster_no = cluster_no
        for _dx, _dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dx = x + _dx
            dy = y + _dy
            if 0 <= dx < self.image_width and 0 <= dy < self.image_height:
                idx = self.index_mesh[self.cluster_matrix == self.cluster_matrix[dy, dx]]
                n_pixel = len(idx)

                if n_pixel == 0 or self.cluster_matrix[dy, dx] == cluster_no:
                    continue
                if n_pixel > biggest_area:
                    biggest_area = n_pixel
                    new_cluster_no = self.cluster_matrix[dy, dx]
        return new_cluster_no

    def enforce_connectivity(self):
        for cluster in self.clusters:
            pixel_set = set(tuple(p) for p in cluster.pixels)
            bfs_queue = queue.Queue()
            if len(pixel_set) == 0:
                continue
            center = (cluster.y, cluster.x)
            if center not in pixel_set:
                np_pixels = np.array(cluster.pixels)
                y_pixels, x_pixels = np_pixels.T
                np_center = np.array(center)
                distances = np.abs(y_pixels - np_center[0]) + np.abs(x_pixels - np_center[1])
                idx = np.argmin(distances)
                center = tuple(np_pixels[idx].tolist())

            bfs_queue.put(center)
            pixel_set.remove(center)

            while bfs_queue.qsize() != 0:
                y, x = bfs_queue.get()
                for _dx, _dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dx = x + _dx
                    dy = y + _dy
                    if 0 <= dx < self.image_width and 0 <= dy < self.image_height:
                        if (dy, dx) in pixel_set:
                            bfs_queue.put((dy, dx))
                            pixel_set.remove((dy, dx))
            bfs_queue.queue.clear()
            while pixel_set:
                pixel = next(iter(pixel_set))
                bfs_queue.put(pixel)
                while bfs_queue.qsize() != 0:
                    y, x = bfs_queue.get()
                    for _dx, _dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        dx = x + _dx
                        dy = y + _dy
                        if 0 <= dx < self.image_width and 0 <= dy < self.image_height:
                            if [dy, dx] not in cluster.pixels:
                                self.cluster_matrix[y, x] = self.cluster_matrix[dy, dx]
                                self.clusters[self.cluster_matrix[dy, dx]].pixels.append([y, x])
                                if [y, x] in cluster.pixels:
                                    cluster.pixels.remove([y, x])
                                bfs_queue.queue.clear()
                                pixel_set.remove(pixel)
                                break
                            else:
                                bfs_queue.put((dy, dx))

    def get_countour_image(self, color=(255, 255, 255)):
        visualize_image = self.image.copy()
        for cluster in self.clusters:
            pixel_mask = self.cluster_matrix == cluster.no
            idx = self.index_mesh[pixel_mask]
            if len(idx) == 0:
                continue
            pixel_mask_uint8 = pixel_mask.astype(np.uint8)
            contours, hierarchy = cv2.findContours(pixel_mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(visualize_image, contours, -1, color, 0)
        return visualize_image

    def calculate_superpixel_image(self):
        cluster_image = self.image.copy()
        for cluster in self.clusters:
            idx = self.cluster_matrix == cluster.no
            if len(self.index_mesh[idx]) == 0:
                continue
            cluster_image[idx] = cluster.get_pixel()
        cluster_image = cv2.cvtColor(cluster_image, cv2.COLOR_LAB2BGR)
        return cluster_image

    def process(self):
        init_clusters_time = time()
        self.__init_clusters()
        init_clusters_time = time() - init_clusters_time
        print(f"Initialize clusters take: {round(init_clusters_time, 2)}s")
        clustering_time = time()
        for _ in trange(self.iter):
            self.clustering()
        clustering_time = time() - clustering_time
        print(f"Clustering {self.iter} times take: {round(clustering_time, 2)}s")
        print("Enforce connectivity")
        enforce_connectivity_time = time()
        self.enforce_connectivity()
        enforce_connectivity_time = time() - enforce_connectivity_time
        print(f"Enforce connectivity take: {round(enforce_connectivity_time, 2)}s")
        visualize_image = self.get_countour_image((0, 0, 0))
        cluster_image = self.calculate_superpixel_image()
        cv2.imshow("window 1", visualize_image)
        cv2.imshow("window 2", cluster_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    image = cv2.imread("lenna.png")
    image = cv2.resize(image, (512, 512))
    slic = SLIC(image, 1000, 10, iter=4)
    slic.process()
