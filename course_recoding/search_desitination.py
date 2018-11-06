from matplotlib.pyplot import show
import networkx


BJ = 'Beijing'
SZ = 'Shenzhen'
GZ = 'Guangzhou'
WH = 'Wuhan'
HLG = 'Heilongjiang'
NY = 'New York City'
CM = 'Chiangmai'
SG = 'Singapore'

air_route = {
    BJ: {SZ, GZ, WH, HLG, NY},
    GZ: {WH, BJ, CM, SG},
    SZ: {BJ, SG},
    WH: {BJ, GZ},
    HLG: {BJ},
    CM: {GZ},
    NY: {BJ}
}

# show net work
# air_route = networkx.Graph(air_route)
#
# networkx.draw(air_route, with_labels=True)
# show()


def search_desitination(graph, start, destination):
    pathes = [[start]]
    arrived = set()
    routine = []
    while pathes:
        path = pathes.pop(0)
        froniter = path[-1]
        if froniter in arrived:
            continue
        # get new lines

        for city in graph[froniter]:
            new_path = path + [city]
            pathes.append(new_path)
            if city == destination:
                return new_path
        arrived.add(city)
    return routine


def draw_routines(routines):
    return '✈ ->'.join(routines)


print(draw_routines(search_desitination(air_route, SZ, CM)))
