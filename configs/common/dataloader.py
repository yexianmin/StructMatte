from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler

from data import ImageFileTrain, DataGenerator

#Dataloader
train_dataset = L(DataGenerator)(
    data = L(ImageFileTrain)(
        alpha_dir='/public/home/tm_yxm/dataset/com-1k/mask',
        fg_dir='/public/home/tm_yxm/dataset/com-1k/fg',
        bg_dir='/public/home/tm_yxm/dataset/com-1k/train2014',
        root='/public/home/tm_yxm/dataset/com-1k'
    ),
    phase = 'train'
)

# /public/home/tm_yxm/dataset/com-1k/
# 
dataloader = OmegaConf.create()
dataloader.train = L(DataLoader)(
    dataset = train_dataset,
    batch_size=15,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=L(DistributedSampler)(
        dataset = train_dataset,
    ),
    drop_last=True
)