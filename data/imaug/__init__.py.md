# `ppocr/data/imaug/__init__.py`代码解析

## transform数据增强函数
```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6.4,6.4)) #调整显示窗口尺寸大小640x640
plt.gca().invert_yaxis() #把y轴箭头朝向改为向下，如果使用的是imshow函数则无需执行
plt.plot(padded_polygon[:,0],padded_polygon[:,1])
plt.plot(np.append(polygon[:,0],polygon[0,0]),np.append(polygon[:,1],polygon[0,1]))
plt.show()
```