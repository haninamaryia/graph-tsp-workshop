import pandas as pd
import math


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

        data = pd.read_csv("country-capitals.csv")
        data = data.drop(columns=['CountryCode'])

        # Create vertices
        for index, row in data.iterrows():
            self.addVertex(row['ContinentName'], row['CountryName'], \
                                    row['CapitalName'], row['CapitalLatitude'],
                                    row['CapitalLongitude'])

        # Add edges
        for v1 in self.vertList:
            for v2 in self.vertList:
                if not v1 == v2:
                    if not self.edgeExists(v1, v2):
                        dist = self.getDistance(v1, v2)
                        self.addEdge(v1, v2, dist)
                        self.addEdge(v2, v1, dist)


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

        if f not in self.vertList or t not in self.vertList:
            raise Exception(f'either {f} or {t} are not in vertList.')

        self.vertList[f].addNeighbor(self.vertList[t], weight)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())