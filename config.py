from easydict import EasyDict as edict

__c=edict()
cfg=__c
__c.train=edict()


#批量大小
__c.train.batch_size=64
#資料集位置
__c.train.dataset_path="dataset/"
#設置驗證集之比例
__c.train.proportion=0.1
#類別數量
__c.train.classes=1501
#隨機種子碼
__c.train.seed=1234
#放置ckpt的位置
__c.train.log_path="AY2OG4"
