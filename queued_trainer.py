# vim: expandtab:ts=4:sw=4
import string
import os
import threading
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
import shutil

def random_sample_identities_forever(batch_size, num_samples_per_id, data_x,
                                     data_y):
    """生成器，每次可隨機生成長度為"batch_size"的data_x,data_y

    Parameters
    ----------
    batch_size : int
        批量大小。
    num_samples_per_id : int
        每個ID有幾個樣本，"num_samples_per_id"必須要能整除"batch_size"否則顯示錯誤。
    data_x : List[string]
        圖片路徑列表
    data_y : List[int]
        label列表
    Returns
    -------
    List[np.ndarray]
        每次回傳長度為"batch_size"的data_x,data_y

    """
    assert (batch_size) % num_samples_per_id == 0
    #決定1個batch有幾個id
    num_ids_per_batch = int((batch_size) / num_samples_per_id)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    #有哪幾類行人
    unique_y = np.unique(data_y[data_y >= 0])
    #dictionary:{類別:[]哪幾個位置為該類別}
    y_to_idx = {y: np.where(data_y == y)[0] for y in unique_y}

    while True:
        #選哪幾群人
        indices = np.random.choice(
            len(unique_y), num_ids_per_batch, replace=False)
        #知道選哪幾群人
        batch_unique_y = unique_y[indices]              
        #[batch_size,len(data_x)]batch_size 型態為String
        batch_x = np.zeros((batch_size, ) + data_x.shape[1:], data_x.dtype)
        #[batch_size,] 型態為int32
        batch_y = np.zeros((batch_size, ), data_y.dtype)                        
        e = 0
        #補充batch_unique_y的batch行人資料
        #若有缺則在下面補
        for i, y in enumerate(batch_unique_y):
            #num_samples_per_id比該編號行人個數
            num_samples = min(num_samples_per_id, len(y_to_idx[y]))
            #在y類別的路徑中找num_sample個例子
            indices = np.random.choice(y_to_idx[y], num_samples, replace=False)
            #設定開始點及結束點
            s, e = e, e + num_samples
            batch_x[s:e] = data_x[indices]
            batch_y[s:e] = y

        # If we need to add more data, random sample ids until we have reached
        # the batch size.
        #直到樣本補完持續增加資料
        num_samples = len(batch_x) - e              
        num_tries = 0
        while num_samples > 0 and num_tries < 100:
            y = np.random.choice(unique_y)
            if y in batch_unique_y:
                # Find a target that we have not yet in this batch.
                #如果batch中已有該類別
                num_tries += 1
                continue

            num_samples = min(num_samples, len(y_to_idx[y]))
            indices = np.random.choice(y_to_idx[y], num_samples, replace=False)
            s, e = e, e + num_samples
            batch_x[s:e] = data_x[indices]
            batch_y[s:e] = y
            #看還差幾個樣本
            num_samples = len(batch_x) - e          

        if e < batch_size:
            print("ERROR: Failed to sample a full batch. Adding corrupt data.")
        yield [batch_x, batch_y]
def _generate_run_id(size=6, chars=None):
    """隨機生成長度為"size"的ID

    Parameters
    ----------
    size : int
    chars : Optional[str]
        生成ID的char

    Returns
    -------
    str
        回傳隨機生成的ID

    """
    if chars is None:
        chars = string.ascii_uppercase + string.digits
    import random
    return ''.join(random.choice(chars) for _ in range(size))


class ThreadSafeIterator(object):
    """
    將迭代器或生成器轉換為多線程，並且一個時間只有一個多線程被給予權限

    Parameters
    ----------
    iterator_or_generator
        An iterator or generator to be wrapped.

    """

    def __init__(self, iterator_or_generator):
        self._iterator_or_generator = iterator_or_generator
        self._lock = threading.Lock()

    def __iter__(self):
    #利用__iter__使生成器or迭代器轉換為可迭代對象就能用for來處理
        return self
    #python3
    def __next__(self):
        '''
        每當迭代時，就會執行此行
        '''
        with self._lock:                            #包括了 acquire()以及release()
            return next(self._iterator_or_generator)
    #python2
    def next(self):
        with self._lock:                            #包括了 acquire()以及release()
            return self._iterator_or_generator.next()


class QueuedTrainer(object):
    """
    用來執行訓練以及評估TensorFlow模型。使用tf.FIFOQueue來同時執行載入圖片以及預處理。

    Parameters
    ----------
    enqueue_vars List[tf.Tensor] [[None,batch_size,64,3],[None,]]:
        包括了被預處裡的路徑以及label
    input_vars Optional[List[tf.Tensor]] [[None,],[None,]]:
        對應"enqueue_vars"的List，包括了圖片路徑以及label
    num_enqueue_threads : Optional[int]
        用於平行預處理數據的線程數
    queue_capacity : Optional[int]
        佇列中的最大元素數；默認為512。
    """

    def __init__(self, enqueue_vars, input_vars=None, num_enqueue_threads=4,
                 queue_capacity=512):
        if input_vars is None:
            input_vars = enqueue_vars
        self._input_vars = input_vars
        self._enqueue_vars = enqueue_vars

        shapes = [var.get_shape().as_list()[1:] for var in enqueue_vars]
        dtypes = [var.dtype for var in enqueue_vars]
        #存放資料
        self._queue = tf.FIFOQueue(queue_capacity, dtypes, shapes)                  
        #幾個多線程
        self._num_enqueue_threads = num_enqueue_threads
        #將多線程存進_enqueue_threads中                             
        self._enqueue_threads = []
        #放進_queue  一次放進batch_size個                                               
        self._enqueue_op = self._queue.enqueue_many(self._enqueue_vars)
        #關閉佇列             
        self._stop_op = self._queue.close(cancel_pending_enqueues=True)             
        self._coordinator = None

        self._feed_generator = None
        self._batch_size = None
        self._init_fns = []

    def get_input_vars(self, batch_size):
        """取前batch_size筆資料

        Parameters
        ----------
        batch_size : int
            The batch size.

        Returns
        -------
        List[tf.Tensor]
            回傳前batch_size筆資料
        """
        self._batch_size = batch_size
        return self._queue.dequeue_many(batch_size)

    def run(self, feed_generator, train_op, log_dir="/tmp/slim_trainer/",
            run_id=None,max_checkpoints_to_keep=0, **kwargs):
        """ Run training.

        Parameters
        ----------
        feed_generator : Iterator[ndarray, ...]
            Generator將其與多執行序結合
        train_op tf.Tensor:
            'slim.learning.create_train_op'
        log_dir : Optional[str]
            放置ckpt以及summaries的位置
        run_id : Optional[str]
            放置ckpt以及summaries的位置，若沒有指定則隨機生成
        max_checkpoints_to_keep : int
            除存最后"max_checkpoints_to_keep"的ckpt，若为0则保存全部ckpt
        kwargs:
            给予"tf.slim.learning.train"的实参
            e.g., "number_of_steps=100"执行100次的迭代

        """
        #宣告生成器
        self._feed_generator = ThreadSafeIterator(feed_generator)
        #宣告協調器(多執行序)
        self._coordinator = tf.train.Coordinator()

        if run_id is None:
            run_id = _generate_run_id(6)
        #儲存之資料夾
        log_dir = os.path.join(log_dir, run_id)
        print("---------------------------------------")
        print("Run ID: ", run_id)
        print("Log directory: ", log_dir)
        print("---------------------------------------")
        #設定要儲存ckpt的數量
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        '''
        the slim.learning.train function does the following:

        For each iteration, evaluate the train_op, which updates the parameters using the optimizer applied to the current minibatch. Also, update the global_step.
        Occasionally store the model checkpoint in the specified directory. This is useful in case your machine crashes - then you can simply restart from the specified checkpoint.
        '''
        #執行train_op、_train_step_fn  用于计算损失和梯度步骤。
        try:                                                                
            slim.learning.train(                                                                                    
                train_op, log_dir, self._train_step_fn, saver=saver,
                **kwargs)
        except UnboundLocalError:
            pass
        #所有迭代執行完把多執行緒的queue清掉
        self._wait_for_threads()
    def _train_step_fn(self, session, train_op, global_step,
                       train_step_kwargs):
        '''
        執行補充資料的多線程工作
        '''
        #若多線程沒有工作
        if len(self._enqueue_threads) == 0:         
            for fn in self._init_fns:
                fn(session)
            #在enque_threads中補充並執行多線程
            self._start_enqueue(session)
        #判斷是否停止訓練
        total_loss, should_stop = slim.learning.train_step(
            session, train_op, global_step, train_step_kwargs)
        if should_stop or self._coordinator.should_stop():
            self._stop_all_threads(session)
        return total_loss, should_stop

    def _stop_all_threads(self, session):
        self._coordinator.request_stop()
        session.run(self._stop_op)  # Close the data_queue.

    def _wait_for_threads(self):
        self._coordinator.join(self._enqueue_threads)
        self._enqueue_threads = []
    def _start_enqueue(self, session, num_threads=None):
        '''
        進行多線程
        '''                
        if num_threads is None:
            num_threads = self._num_enqueue_threads
        for _ in range(num_threads):                                    
            thread = threading.Thread(                                  #多線程做target的工作
                target=self._run_enqueue_thread, args=(session, ))
            thread.start()
            self._enqueue_threads.append(thread)                        #thread放入_enqueue_threads

    def _run_enqueue_thread(self, session):
        try:
            #獲取每一個batch_size資料
            for data in self._feed_generator:
                if self._coordinator.should_stop():
                    break
                try:
                    #_input_vars的placeholder對上data
                    feed_dict = {
                        var: value for var, value in
                        zip(self._input_vars, data)}
                    #feed_dict會經過處理傳入enque
                    session.run(self._enqueue_op, feed_dict=feed_dict)
                except (tf.errors.CancelledError, tf.errors.AbortedError):
                    # We have been requested to stop enqueuing data.
                    break
        except Exception as e:
            print("EnqueueError:", e)
            self._stop_all_threads(session)