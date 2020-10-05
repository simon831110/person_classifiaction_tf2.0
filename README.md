# person_classification
# 介紹
這是使用卷積神經網路建置的行人分類的code
# 建置環境
Python 3<br />
TensorFlow>=1.9.0<br />
numpy<br />
os<br />
# 數據集
數據集包含了Market1501以及MARS下載路徑為:http://www.liangzheng.com.cn/Project/project_reid.html ，並將行人圖片依照ID分布在不同的資料夾中<br />
<pre>
/dataset<br />
    /dataset/0<br />
    /dataset/1<br />
    /dataset/2<br />
    .<br />
    .<br />
    .<br />
</pre>
# 訓練
先去**config.py**修改相關配置
