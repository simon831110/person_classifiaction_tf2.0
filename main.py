# vim: expandtab:ts=4:sw=4
from config import cfg
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import functools
import os
import queued_trainer
from network import network_definition



def train_dir_to_str(directory):
    '''
    將訓練集轉換為
        1.圖片名稱
        2.圖片相對號碼,例:[1,1,1,2,2,2]
        3.圖片對應拍攝鏡頭
    Parameter
    ---------
    directory:
        資料集之位置
    Returns
    -------
    image_filenames List(str):
        圖片名稱(全路徑)
    ids List(int):
        圖片相對號碼,例:[1,1,1,2,2,2]
    camera_indices List(int):
        圖片對應拍攝鏡頭
    '''

    def to_label(x):
        return int(x) if x.isdigit() else -1

    dirnames = os.listdir(directory)
    image_filenames, ids, camera_indices = [], [], []
    for dirname in dirnames:
        filenames = os.listdir(os.path.join(directory, dirname))
        filenames = [
            f for f in filenames if os.path.splitext(f)[1] == ".jpg"]
        image_filenames += [
            os.path.join(directory, dirname, f) for f in filenames]
        ids += [to_label(dirname) for _ in filenames]
        camera_indices += [int(f[5]) for f in filenames]

    return image_filenames, ids, camera_indices


def split_validation(data_y, num_validation_y, seed=None):
    '''
    將訓練集切割驗證集(依照ID分1/10為驗證其餘為訓練)
    Parameters
    ----------
    data_y List[int]:
        資料集有哪些ID
    num_validation_y float:
        資料集有幾%的驗證集
    seed int:
        隨機點
    Returns
    -------
    np.where(training_mask)[0] List[int]:
        訓練集的位置
    np.where(validation_mask)[0] List[int]:
        驗證集的位置
    '''
    unique_y = np.unique(data_y)
    #將num_validation_y轉成整數
    if isinstance(num_validation_y, float):         
        num_validation_y = int(num_validation_y * len(unique_y))
    
    random_generator = np.random.RandomState(seed=seed)
    validation_y = random_generator.choice(                                 #從unique_y中隨機找出num_validation_y個驗證人次
        unique_y, num_validation_y, replace=False)

    validation_mask = np.full((len(data_y), ), False, bool)

    for y in validation_y:
        validation_mask = np.logical_or(validation_mask, data_y == y)
    training_mask = np.logical_not(validation_mask)
    return np.where(training_mask)[0], np.where(validation_mask)[0]

class data_load(object):

    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        '''
        Parameters
        ----------
        dataset_dir:
            資料集的位置
        num_validation_y:
            驗證集之比例
        seed:
            種子碼
        '''
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed

    def read_train(self):
        '''
        Returns
        -------
        回傳訓練集之資料
        filenames:List[str]
            訓練集檔案路徑       
        ids:List[int]
            訓練集ID名稱
        camera_indices:List[int]
            訓練集由第幾架Camera拍攝
        '''
        filenames, ids, camera_indices = train_dir_to_str(
            self._dataset_dir)
        train_indices, _ = split_validation(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in train_indices]
        ids = [ids[i] for i in train_indices]
        camera_indices = [camera_indices[i] for i in  train_indices]
        return filenames, ids, camera_indices

dataset = data_load(cfg.train.dataset_path, num_validation_y=cfg.train.proportion, seed=cfg.train.seed)

train_x, train_y,_ = dataset.read_train()
print(len(train_x),len(train_y))

network=network_definition.create_network_fn(is_training=True, num_classes=cfg.train.classes, weight_decay=1e-8, reuse=None)

######################################################################################################################
                                   #訓練階段
######################################################################################################################
def _create_softmax_loss(feature_var, logit_var, label_var):
    '''
    計算損失以及辨識準確率
    Parameters
    ----------
    feature_var batch_size*[1*128]tf.float:
        神經網路輸出的特徵
    logit_var batch_size*[1*num_classes]tf.int:
        對應類別的特徵
    label_var [1*128]int:
        對應之label
    '''
    del feature_var  # Unused variable
    cross_entropy_var = slim.losses.sparse_softmax_cross_entropy(
        logit_var, tf.cast(label_var, tf.int64))
    tf.summary.scalar("cross_entropy_loss", cross_entropy_var)

    accuracy_var = slim.metrics.accuracy(
        tf.cast(tf.argmax(logit_var, 1), tf.int64), label_var)
    tf.summary.scalar("classification accuracy", accuracy_var)

def _create_loss(
        feature_var, logit_var, label_var):
    '''
    Parameters
    ----------
    feature_var batch_size*[1*128]tf.float:
        神經網路輸出的特徵
    logit_var batch_size*[1*num_classes]tf.int:
        對應類別的特徵
    label_var [1*128]int:
        對應之label
    '''
    _create_softmax_loss(feature_var, logit_var, label_var)
def preprocess(image, is_training=False, input_is_bgr=False):
    #包括normalization以及預處理
    if input_is_bgr:
        image = image[:, :, ::-1]
    image = tf.divide(tf.cast(image, tf.float32), 255.0)
    if is_training:
        image = tf.image.random_flip_left_right(image)

    return image
def train_loop(preprocess, network, train_x, train_y,
               num_images_per_id, batch_size=128, log_dir='log',
               learning_rate=1e-5,save_summaries_secs=60,
               save_interval_secs=300):
    '''
    執行訓練
    Parameters
    ----------
    preprocess:
        預處理函式傳入圖片編碼方式為tf.uint8傳回圖片編碼方式為tf.float32並進行預處理
    network:
        神經網路架構函式回傳特徵
    train_x List[1*num_data]  str:
        訓練圖片之位置
    train_y List[1*num_data] int:
        訓練圖片之對應label
    num_images_per_id int:
        每次訓練每個label取幾張圖片。batch_size / num_images_per_id 必須被整除。
    batch_size int:
        每次訓練迭代取幾張圖片
    log_dir str:
        儲存訓練檔之位置
    image_shape Tuple[int, int, int]:
        圖片的shape(高度,寬度,通道數)。
    learning_rate float:
        自適應的學習率預設為1e-3
    save_summaries_secs int:
        tensorboard每次更新時間為多長(秒)
    save_interval_secs int:
        每幾秒寫入一次ckpt
    '''
    trainer, train_op = create_trainer(preprocess, network, (128,64,3), batch_size,
                                       learning_rate=learning_rate)
    #generator 生成batch_x以及batch_y
    feed_generator = queued_trainer.random_sample_identities_forever(
        batch_size, num_images_per_id, train_x, train_y)                        
    #儲存變數
    variables_to_restore = slim.get_variables_to_restore(
        exclude=None)
    #執行訓練
    trainer.run(
        feed_generator, train_op, log_dir, 
        run_id=cfg.train.log_path, save_summaries_secs=save_summaries_secs,
        save_interval_secs=save_interval_secs, number_of_steps=None)
    
def create_trainer(preprocess, network, image_shape, batch_size, learning_rate=1e-3):
    '''
    代入多執行緒的訓練器以及以佇列載入資料
    Parameters
    ----------
    preprocess:
        預處理函式傳入圖片編碼方式為tf.uint8傳回圖片編碼方式為tf.float32並進行預處理
    network:
        神經網路架構函式回傳特徵
    image_shape Tuple[int, int, int]:
        圖片的shape(高度,寬度,通道數)。
    batch_size int:
        每次訓練迭代取幾張圖片
    learning_rate float:
        自適應的學習率預設為1e-3
    Return
    ------
    trainer:object
        queued_trainer.QueuedTrainer
    train_op:
    '''
    num_channels=3
    #設置placeholder置放label變數
    label_var = tf.placeholder(tf.int64, (None,))      
    #設置placeholder置放圖片路徑
    filename_var = tf.placeholder(tf.string, (None, ))               
    #將圖片路徑轉化為Tensor[uint8]
    image_var = tf.map_fn(                                              
        lambda x: tf.image.decode_jpeg(
            tf.read_file(x), channels=num_channels),
        filename_var, back_prop=False, dtype=tf.uint8)
    #resize
    image_var = tf.image.resize_images(image_var, image_shape[:2])
    #[tf.placeholder(圖片路徑),tf.placeholder(label)]
    input_vars = [filename_var, label_var]
    #[Tensor(預處理後的圖片),tf.placeholder(label)]
    enqueue_vars = [
        tf.map_fn(
            lambda x: preprocess(x, is_training=True),
            image_var, back_prop=False, dtype=tf.float32),
        label_var]

    #宣告佇列訓練器
    trainer = queued_trainer.QueuedTrainer(enqueue_vars, input_vars)
    #傳回batch_size筆資料
    image_var, label_var = trainer.get_input_vars(batch_size)
    #顯示3張batch中的圖片
    tf.summary.image("images", image_var)
    #將圖片傳入神經網路回傳batch_size*[1*128]、batch_size*[1*num_classes]
    feature_var, logit_var = network(image_var)
    #計算損失
    _create_loss(feature_var, logit_var, label_var)

    variables_to_train = tf.trainable_variables()       
        
    #知道是第幾次迭代
    global_step = tf.train.get_or_create_global_step()

    loss_var = tf.losses.get_total_loss()
    #创建train_op執行優化內有損失、優化方案......並回傳
    train_op = slim.learning.create_train_op(                       
        loss_var, tf.train.AdamOptimizer(learning_rate=learning_rate),
        global_step, summarize_gradients=False,
        variables_to_train=variables_to_train)
    tf.summary.scalar("total_loss", loss_var)
    tf.summary.scalar("learning_rate", learning_rate)

    regularization_var = tf.reduce_sum(tf.losses.get_regularization_loss())
    tf.summary.scalar("weight_loss", regularization_var)
    return trainer, train_op
        
        
train_loop(preprocess, network, train_x, train_y, num_images_per_id=4)
