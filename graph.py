import pandas as pd
import math, sys, numpy as np
from collections import deque, namedtuple

class Vertex:
    def __init__(self, continent, country, city, latitude, longitude):
        self.id = city
        self.connectedTo = {}
        self.continent = continent
        self.country = country
        self.latitude = latitude
        self.longitude = longitude
        self.visitedOnce = False
        self.visitedTwice = False

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self,nbr):
        return self.connectedTo[nbr]

class Edge:
    def __init__(self,city1, city2, distance):
        self.distance = distance
        self.city1 = city1
        self.city2 = city2
        self.rightChild = None
        self.leftChild = None

    def insert(self, city1, city2, distance):
        if (self.city1 == city1 and self.city2 == city2) \
                or (self.city1 == city2 and self.city2 == city1):
            return False

        if city1 == city2:
            return False

        elif self.distance > distance:
            if self.leftChild:
                return self.leftChild.insert(city1, city2, distance)
            else:
                self.leftChild = Edge(city1, city2, distance)
                return True
        else:
            if self.rightChild:
                return self.rightChild.insert(city1, city2, distance)
            else:
                self.rightChild = Edge(city1, city2, distance)
                return True

class EdgeTree:
    def __init__(self):
        self.root = None
        self.size = 0
        self.sortedEdges = []

    def insert(self, city1, city2, distance):
        if self.root:
            return self.root.insert(city1, city2, distance)
        else:
            self.root = Edge(city1, city2, distance)
            return True

    def findMin(self):
        current = self.root
        while current.leftChild:
            current = current.leftChild
        return current.value

    def DFS(self, root):
        if not root:
            return False
        self.DFS(root.leftChild)
        self.sortedEdges.append({"city1": root.city1, "city2" :root.city2,
                                "distance" :root.distance})
        print(root.distance)
        self.DFS(root.rightChild)



class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, continent, country, city, latitude, longitude):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(continent, country, city, latitude, longitude)
        self.vertList[city] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def edgeExists(self, v1, v2):
        if v1 in self.getVertex(v2).connectedTo.keys():
            return True
        else:
            return False

    def getDistance(self, city1, city2):

        lat1 = self.vertList[city1].latitude
        long1 = self.vertList[city1].longitude

        lat2 = self.vertList[city2].latitude
        long2 = self.vertList[city2].longitude

        lat_dist = lat2 - lat1
        long_dist = long2 - long1

        return math.hypot(lat_dist, long_dist)

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], weight)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

    ### IMPLEMENT get_continent_shortest_path() HERE ###
    # https://www.youtube.com/watch?v=CL1byLngb5Q

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):

        # Initilaize minimum distance for next node
        min = np.Infinity

        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v

        return min_index

    def dijkstra(self, src, dest):

        assert src in self.vertList, 'Such source node doesn\'t exist'

        distances = {vertex: np.Infinity for vertex in self.vertList}
        distances[src] = 0

        previous_vertices = {
            vertex: None for vertex in self.vertList
        }

        vertices = self.vertList.copy()

        while vertices:

            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])

            if distances[current_vertex] == np.Infinity:
                break

            # Find unvisited neighbors for the current node
            # and calculate their distances through the current node.
            for neighbor in self.getVertex(current_vertex).connectedTo:

                cost = self.getVertex(current_vertex).connectedTo[neighbor]
                possible_path = distances[current_vertex] + cost

                if possible_path < distances[neighbor.id]:

                    distances[neighbor.id] = possible_path
                    previous_vertices[neighbor.id] = current_vertex

            del(vertices[current_vertex])

        path, current_vertex = deque(), dest

        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return path


    ### IMPLEMENT get_tsp_tour() HERE ###
    def two_apt(self, tour):
        pass

    # def checkPath(self, path):
    #     edges = dict()
    #     for edge in path:


    def greedyTSP(self):

        # create BST of all edges
        edgeTree = EdgeTree()

        i = 0
        for v in self.vertList:
            for n in self.getVertex(v).connectedTo:
                e = edgeTree.insert(v, n.id, self.getVertex(v).connectedTo[n])
                if e:
                    i = i + 1
        edgeTree.size = i

        # construct greedy path
        path = []
        length = 0

        sorted_edges = edgeTree.DFS(edgeTree.root)

        for edge in edgeTree.sortedEdges:

            if len(path) == capitalsGraph.numVertices -1:
                break

            v1 = capitalsGraph.getVertex(edge['city1'])
            v2 = capitalsGraph.getVertex(edge['city2'])

            if v1.visitedTwice or v2.visitedTwice:
                continue
            elif v1.visitedOnce is False and v2.visitedTwice is False:
                v1.visitedOnce = True
                v2.visitedOnce = True
                path.append(edge)
                length = length + abs(self.getDistance(edge['city1'],
                                                    edge['city2']))

            elif v1.visitedOnce is False and v2.visitedOnce is True:
                v2.visitedTwice = True
                v1.visitedOnce = True
                path.append(edge)
                length = length + abs(self.getDistance(edge['city1'],
                                                    edge['city2']))

            elif v1.visitedOnce is True and v2.visitedOnce is False:
                v1.visitedTwice = True
                v2.visitedOnce= True
                path.append(edge)
                length = length + abs(self.getDistance(edge['city1'],
                                                    edge['city2']))

            elif v1.visitedOnce is True and v2.visitedOnce is True:
                v1.visitedTwice = True
                v2.visitedTwice = True
                path.append(edge)
                length = length + abs(self.getDistance(edge['city1'],
                                                    edge['city2']))


        print('ok')
        #

        pass


if __name__ == '__main__':

    data = pd.read_csv("country-capitals.csv")
    data = data.drop(columns=['CountryCode'])

    capitalsGraph = Graph()

    # Create vertices
    for index, row in data.iterrows():
        capitalsGraph.addVertex(row['ContinentName'], row['CountryName'],\
                                row['CapitalName'], row['CapitalLatitude'],
                                row['CapitalLongitude'])

    # Add edges
    for v1 in capitalsGraph.vertList:
        for v2 in capitalsGraph.vertList:
            if not v1 == v2:
                if not capitalsGraph.edgeExists(v1, v2):
                    dist = capitalsGraph.getDistance(v1, v2)
                    capitalsGraph.addEdge(v1, v2, dist)
                    capitalsGraph.addEdge(v2, v1, dist)

    # Find TSP path for all cities
    #print(capitalsGraph.dijkstra('Minsk', 'Ottawa'))

    capitalsGraph.greedyTSP()

