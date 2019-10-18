import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import random

class Point():
    def __init__(self, x, y, city, country):
        self.x = x
        self.y = y
        self.city = city
        self.country = country
        self.node = {"x0": None,
                     "y0": None,
                     "width": None,
                     "height": None}
        self.closest = None
        self.min_dist = np.inf
        self.visited = False


class Node():
    def __init__(self, x0, y0, w, h, points, countries):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.children = []
        self.points = points
        self.countries = countries
        self.parent = None
        self.visited = False

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_points(self):
        return self.points

class QTree():

    def __init__(self, k):

        self.threshold = k
        self.points = []
        self.path = []
        self.points_visited = 0
        self.path_length = 0

        data = pd.read_csv("worldcities.csv")
        data = data[['city', 'lat', 'lng', 'country']]

        # Create vertices
        min_point = Point(np.inf, np.inf, 'Unknown', 'Unknown')
        max_point = Point(0, 0, 'Unknown', 'Unknown')

        for index, row in data.iterrows():
            self.add_point(row['lng'],row['lat'], row['city'], row['country'])

            if row['lat'] + row['lng'] < min_point.x + min_point.y:
                min_point = Point(row['lng'],row['lat'], row['city'],
                                 row['country'])

            if row['lat'] + row['lng'] > max_point.x + max_point.y:
                max_point = Point(row['lng'],row['lat'], row['city'],
                                 row['country'])


        w = max_point.x - min_point.x
        h = max_point.y - min_point.y
        min_node = Node(min_point.x, min_point.y, w, h, self.points,
                        set([p.country for p in self.points]))

        self.root = min_node

        self.subdivide()

        print('quadtree is subdivided.')
        # self.graph()

    def add_point(self, x, y, city, country):
        self.points.append(Point(x, y, city, country))

    def get_points(self):
        return self.points


    def subdivide(self):
        recursive_subdivide(self.root, self.threshold)

    def graph(self):
        fig = plt.figure(figsize=(12, 8))
        plt.title("Quadtree")
        ax = fig.add_subplot(111)
        c = find_children(self.root)
        print("Number of segments: %d" %len(c))
        areas = set()
        for el in c:
            areas.add(el.width*el.height)
        print("Minimum segment area: %.3f units" %min(areas))
        for n in c:
            ax.add_patch(matplotlib.patches.Rectangle((n.x0, n.y0), n.width,
                                                 n.height,
                                            fill=False))
        x = [point.x for point in self.points]
        y = [point.y for point in self.points]
        plt.plot(x, y, 'ro')
        plt.show()
        return


    def traverse(self, parent, node, point):

        if node.children:
            for child in node.children:
                if child is not None:
                    if (child.x0 <= point.x <= (child.x0 + child.width)) and \
                       (child.y0 <= point.y <= (child.y0 + child.height)):

                        if point.city in [p.city for p in child.points]:
                            parent = node
                            node = child
                            return self.traverse(parent, node, point)
                        else:
                            return node, point
        else:
            return node, point


    def get_node(self, point):

        # Find the point
        node = self.root
        parent = None
        return self.traverse(parent, node, point)

    #TODO: if the point is visited, chose second closest point
    def check_distances(self, node, pt, min_dist, min_point):

        # Cause difference of a negative and positive number gives a bigger
        # positive number ...

        min_pt_idx = -1
        i = 0

        for point in node.points:

            if not point.visited:
                if not point.city == pt.city:
                    x = pt.x + 360
                    y = pt.y + 180
                    x2 = point.x + 360
                    y2 = point.y + 180

                    dist = math.sqrt(math.pow(x2 - x, 2) + math.pow(y2 - y, 2))

                    if dist < min_dist:
                        min_dist = dist
                        min_point = point
                        min_pt_idx = i
            i = i + 1

        if min_pt_idx >= 0:
            node.points[min_pt_idx].visited = True
            self.points_visited = self.points_visited + 1

        return min_point, min_dist



    def touching(self, node1, node2):
        node1_up = node1.y0 + node1.height
        node1_down = node1.y0
        node1_left = node1.x0
        node1_right = node1.x0 + node1.width
        node1_sides = [node1_up, node1_down, node1_left, node1_right]

        node2_up = node2.y0 + node2.height
        node2_down = node2.y0
        node2_left = node2.x0 + node2.width
        node2_right = node2.x0 + node2.width
        node2_sides = [node2_up, node2_down,node2_left, node2_right ]

        for side in node1_sides:
            if side in node2_sides:
                return True
        return False

    #Find the leaf node touching our OG node and check the points inside ...
    def visit_children(self, node, pt_node, point):

            if node.visited is False:
                node.visited = True
                if not node == pt_node:
                    point.closest, point.min_dist = self.check_distances(node,
                                        point,point.min_dist,point.closest)

                if node.children:
                    if self.touching(pt_node, node.children[0]):
                        return self.visit_children(node.children[0], pt_node, point)

                    if self.touching(pt_node, node.children[1]):
                        return self.visit_children(node.children[1], pt_node, point)

                    if self.touching(pt_node, node.children[2]):
                        return self.visit_children(node.children[2], pt_node, point)

                    if self.touching(pt_node, node.children[3]):
                        return self.visit_children(node.children[3], pt_node, point)


    def find_closest_neighbour(self, node, point):

        pt_node = node

        point.closest, point.min_dist = self.check_distances(node, point,
                                                          point.min_dist,
                                                             point.closest)

        # Sister nodes, i.e. nodes sharing the same parent as the OG node ...
        while node.parent:
            node = node.parent
            self.visit_children(node, pt_node, point)


    def shortest_path(self, pt):
        pass


# TODO: use integers instead of floats
def recursive_subdivide(node, k):
    if len(node.points) <= k:
        return

    w_ = float(node.width / 2)
    h_ = float(node.height / 2)

    p, c = contains(node.x0, node.y0, w_, h_, node.points)
    x1 = Node(node.x0, node.y0, w_, h_, p, c)
    recursive_subdivide(x1, k)

    p, c = contains(node.x0, node.y0 + h_, w_, h_, node.points)
    x2 = Node(node.x0, node.y0 + h_, w_, h_, p, c)
    recursive_subdivide(x2, k)

    p, c = contains(node.x0 + w_, node.y0, w_, h_, node.points)
    x3 = Node(node.x0 + w_, node.y0, w_, h_, p, c)
    recursive_subdivide(x3, k)

    p, c = contains(node.x0 + w_, node.y0 + w_, w_, h_, node.points)
    x4 = Node(node.x0 + w_, node.y0 + h_, w_, h_, p, c)
    recursive_subdivide(x4, k)

    node.children = [x1, x2, x3, x4]
    x1.parent = node
    x2.parent = node
    x3.parent = node
    x4.parent = node


def contains(x, y, w, h, points):
    pts = []
    countries = []

    for point in points:
        if point.x >= x and point.x <= x + w and point.y >= y and point.y <= \
                y + h:

            pts.append(point)
            point.node = {"x0": x,
                    "y0": y,
                    "width": w,
                    "height": h}
            countries.append(point.country)
    c = set(countries)

    return pts, c


def find_children(node):
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += (find_children(child))
    return children



