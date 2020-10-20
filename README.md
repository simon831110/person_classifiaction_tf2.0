# person_classification
# 介紹
這是使用卷積神經網路建置的行人分類的code
# 建置環境
Python 3<br />
TensorFlow>=2.0.0<br />
numpy<br />
os<br />
# 數據集
數據集包含了Market1501以及MARS下載路徑為:http://www.liangzheng.com.cn/Project/project_reid.html ，並將行人圖片依照ID分布在不同的資料夾中<br />
<pre>
/dataset<br />
    /dataset/0000<br />
    /dataset/0001<br />
    /dataset/0002<br />
    .<br />
    .<br />
    .<br />
</pre>
使用seperate_train_split.ipynb將資料分為訓練集以及測試集
# 訓練
先去self_trainer修改訓練路徑
<pre>
# I commented out some of the code for learning the model.
def main():
    train_dict = reader('dataset_seperate/train/')
    X_train = np.array(train_dict['image'])
    y_train = to_categorical(np.array(train_dict['label']))

    train_model(X_train, y_train)

if __name__=='__main__':
    main()
</pre>
# 測試
先去self_trainer_test修改訓練路徑
<pre>
# I commented out some of the code for learning the model.
def main():
    test_dict = reader('dataset_seperate/test/')
    X_test = np.array(test_dict['image'])
    label = np.array(test_dict['label'])
    # I do not recommend trying to train the model on a kaggle.
    #train_model(X_train, y_train)
    test_model(X_test, label)

if __name__=='__main__':
    main()
</pre>
