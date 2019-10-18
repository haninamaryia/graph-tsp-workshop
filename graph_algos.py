import medium_graph_impl as g
import small_graph_impl as sg
import random

# Used with the small graph in ou worskhop
def greedyTSP(graph):
    # create BST of all edges
    edgeTree = sg.EdgeTree()

    i = 0
    for v in graph.vertList:
        for n in graph.getVertex(v).connectedTo:
            e = edgeTree.insert(v, n.id, graph.getVertex(v).connectedTo[n])
            if e:
                i = i + 1
    edgeTree.size = i

    # construct greedy path
    path = []
    length = 0

    edgeTree.DFS(edgeTree.root)

    for edge in edgeTree.sortedEdges:

        if len(path) == graph.numVertices - 1:
            break

        v1 = graph.getVertex(edge['city1'])
        v2 = graph.getVertex(edge['city2'])

        if v1.visitedTwice or v2.visitedTwice:
            continue
        elif v1.visitedOnce is False and v2.visitedTwice is False:
            v1.visitedOnce = True
            v2.visitedOnce = True
            path.append(edge)
            length = length + abs(graph.getDistance(edge['city1'],
                                                   edge['city2']))

        elif v1.visitedOnce is False and v2.visitedOnce is True:
            v2.visitedTwice = True
            v1.visitedOnce = True
            path.append(edge)
            length = length + abs(graph.getDistance(edge['city1'],
                                                   edge['city2']))

        elif v1.visitedOnce is True and v2.visitedOnce is False:
            v1.visitedTwice = True
            v2.visitedOnce = True
            path.append(edge)
            length = length + abs(graph.getDistance(edge['city1'],
                                                   edge['city2']))

        elif v1.visitedOnce is True and v2.visitedOnce is True:
            v1.visitedTwice = True
            v2.visitedTwice = True
            path.append(edge)
            length = length + abs(graph.getDistance(edge['city1'],
                                                   edge['city2']))
    return path


def two_opt(graph, tour):
    pass

# Used with the medium graph in our workshop
def nearestNeighbor(graph):
    pass

if __name__ == '__main__':


    smallgraph = sg.Graph()
    path = greedyTSP(smallgraph)

    ### TODO: sort the edges in order to get a a -> b , b -> c type of
    ###  printed string


    #########################################################

    tree = g.QTree(1)
    node, point = tree.get_node(g.Point(4.1800, 7.6300, 'Iwo',
                                                'Nigeria'))
    tree.find_closest_neighbour(node, point)
    print('done')

    tree.graph()

    start_pt = random.choice(tree.points)
    start_node = tree.get_node(start_pt)
    tree.path.append(start_pt)
    tree.shortest_path(start_pt)
    print('path finished.')
