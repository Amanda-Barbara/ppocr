# operators.py代码解析

## NormalizeImage
* `original after EastRandomCrop`

![](images/original_img_after_EastRandomCrop.png)

* `after NormalizeImage`

## KeepKeys
* 返回`['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']`
键值
  
![](images/after_normalize.png)

## ToCHWImage改变图像数组排序
```python
data['image'] = img.transpose((2, 0, 1)) # HWC->CHW
```
* numpy查看多维数组的结构顺序，按照最后的两个维度看起，  
  比如HWC三个维度，最后两个维度WC表示，有W行C列，加上H表示  
  有H个W行C列的二维数组，