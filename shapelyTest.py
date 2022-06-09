import numpy as np
from matplotlib import pyplot
from shapely.geometry import Point
from descartes import PolygonPatch

BLUE = 'blue'
GRAY = 'gray'

fig = pyplot.figure()

a = Point(1, 1).buffer(1.5)
b = Point(1.5, 1).buffer(2)

# 1
ax = fig.add_subplot(121)

patch1 = PolygonPatch(a, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
ax.add_patch(patch1)
patch2 = PolygonPatch(b, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
ax.add_patch(patch2)
c = a.difference(b)
patchc = PolygonPatch(c, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patchc)
# c = a.union(b)
x, y = c.exterior.coords.xy
print(x)
print(y)
pyplot.plot(x, y, color="red", linewidth=0.5, linestyle="-", label="circle")
ax.set_title('a.difference(b)')
pyplot.xlim([-1, 4])
pyplot.ylim([-1, 3])
ax.set_aspect(1) # x,y坐标轴刻度等比例显示

#2
ax = fig.add_subplot(122)

patch1 = PolygonPatch(a, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
ax.add_patch(patch1)
patch2 = PolygonPatch(b, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
ax.add_patch(patch2)
c = b.difference(a)
patchc = PolygonPatch(c, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patchc)
x, y = c.exterior.coords.xy
print(x)
print(y)
pyplot.plot(x, y, color="red", linewidth=0.5, linestyle="-", label="circle")
ax.set_title('b.difference(a)')
pyplot.xlim([-1, 4])
pyplot.ylim([-1, 3])
ax.set_aspect(1)
pyplot.show()
