import networkx
import matplotlib


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
air_route = networkx.Graph(air_route)

networkx.draw(air_route, with_labels=True)
