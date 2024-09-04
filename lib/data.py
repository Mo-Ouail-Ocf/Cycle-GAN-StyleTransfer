# data source : https://www.kaggle.com/datasets/suyashdamle/cyclegan

from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset,DataLoader
from ignite.handlers.tensorboard_logger import TensorboardLogger
from torchvision.utils import make_grid
import numpy as np
data_path = Path(__file__).parent / 'data' / 'vangogh2photo' / 'vangogh2photo'

normal_path_train = data_path / 'trainB'
style_path_train = data_path / 'trainA'
normal_path_test = data_path / 'testB'
style_path_test = data_path / 'testA'

def get_imgs_paths(normal_path:Path,style_path:Path,test=False)->list[str]:

    normal_files_paths = normal_path.glob('*.jpg')
    style_files_paths = style_path.glob('*.jpg')

    normal_files_paths = [str(file) for file in normal_files_paths]
    style_files_paths = [str(file) for file in style_files_paths]

    if test:
        style_files_paths = style_files_paths[:16]
        normal_files_paths = normal_files_paths[:16]
        
    return normal_files_paths,style_files_paths

class VangoghToPhotoDataSet(Dataset):
    def __init__(self,normal_files_paths:list[str],
                 style_files_paths:list[str],transform):
        self.normal_files_paths =normal_files_paths 
        self.style_files_paths =style_files_paths 
        self.transform = transform
        self.normal_len = len(normal_files_paths) 
        self.style_len = len(style_files_paths) 
        self.ds_len = max(self.normal_len,self.style_len)

    def __len__(self):
        return self.ds_len

    def __getitem__(self, index):
        style_path = self.style_files_paths[index % self.style_len]
        normal_path = self.normal_files_paths[index % self.normal_len]

        normal_img , style_img = Image.open(normal_path),Image.open(style_path)
        normal_img , style_img =np.asarray(normal_img,dtype=np.uint8) ,np.asarray(style_img,dtype=np.uint8)
        transform_imgs = self.transform(image=normal_img,styled_img=style_img)

        normal_img_t , style_img_t = transform_imgs['image'],transform_imgs['styled_img']

        return normal_img_t.to('cuda'),style_img_t.to('cuda')



train_normal_files_paths,train_style_files_paths = get_imgs_paths(normal_path_train,style_path_train)
test_normal_files_paths,test_style_files_paths = get_imgs_paths(normal_path_test,style_path_test)

mean , std = [0.5,0.5,0.5],[0.5,0.5,0.5]
transform_train = A.Compose(
    [
        A.Resize(height=256,width=256),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5), 
        A.Normalize(mean=mean,std=std,max_pixel_value=255),
        ToTensorV2()
    ],
    additional_targets={
        'styled_img':'image'
    }
)
transform_valid = A.Compose(
    [
        A.Resize(height=256,width=256),
        A.Normalize(mean=mean,std=std,max_pixel_value=255),
        ToTensorV2()
    ],
    additional_targets={
        'styled_img':'image'
    }
)

train_ds = VangoghToPhotoDataSet(normal_files_paths=train_normal_files_paths,
                                 style_files_paths=train_style_files_paths,
                                 transform=transform_train)

valid_ds = VangoghToPhotoDataSet(normal_files_paths=test_normal_files_paths,
                                 style_files_paths=test_style_files_paths,
                                 transform=transform_valid)



train_dl = DataLoader(train_ds,batch_size=8,shuffle=True,drop_last=True)
valid_dl = DataLoader(valid_ds,batch_size=8,shuffle=True,drop_last=True)


if __name__=="__main__":
    batch = next(iter(train_dl))
    input_images, real_output_images = batch

    input_images = input_images.cpu().detach()
    real_output_images = real_output_images.cpu().detach()


    input_grid = make_grid(input_images, nrow=4, normalize=True, value_range=(-1, 1))
    output_grid = make_grid(real_output_images, nrow=4, normalize=True, value_range=(-1, 1))

    
    tb_logger = TensorboardLogger('./log_imgs_test')

    # Log the images
    tb_logger.writer.add_image("Inputs", input_grid, 0)
    tb_logger.writer.add_image("Outputs", output_grid, 0)
    tb_logger.close()
