a = [[10,9,6,3,7],
 [6,10,2,9,7],
 [7,6,3,8,2],
 [8,9,7,9,9],
 [6,8,6,8,2]]

columns = len(a)
rows = len(a[0])

_m_return = []

for r in range(0, rows):
    line = []
    for c in range(columns):
        line.append(a[rows-(c+1)][r])
    _m_return.append(line)

print(_m_return)