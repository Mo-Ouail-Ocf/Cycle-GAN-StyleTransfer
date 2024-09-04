from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine import Engine,Events
from torchvision.utils import make_grid 
from ignite.contrib.handlers import tqdm_logger
import torch.nn as nn
from .data import valid_dl
import torch
from ignite.handlers.checkpoint import Checkpoint,DiskSaver
from config.config import Config

LOG_IMAGES_EVERY_N_EPOCH = 5

def attach_checkpoint_handler(engine: Engine, models: dict, optimizers: dict):
    checkpoint_handler = Checkpoint(
        to_save={**models, **optimizers, 'engine': engine},
        save_handler=DiskSaver(
            dirname=Config.model_folder,
            create_dir=True,
            require_empty=False
        ),
        n_saved=2,
        filename_prefix='model',
        global_step_transform=lambda engine, event: engine.state.epoch
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

def attach_ignite(
        trainer:Engine,
        style_generator:nn.Module,
        nature_generator:nn.Module,
):
    
    tb_logger = TensorboardLogger(log_dir ='./cycle_gan_log')

    tqdm_train = tqdm_logger.ProgressBar().attach(trainer,output_transform=lambda x:x)

    
    tb_logger.attach_output_handler(
        engine=trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag='train',
        output_transform=lambda x: x
    )

    @torch.no_grad
    def log_generated_images(engine, logger, style_generator,nature_generator, epoch):
        style_generator.eval()
        nature_generator.eval()
        
        batch = next(iter(valid_dl))
        input_images, styled_images = batch

        # Generate fake images
        styled_snyth_imgs = style_generator(input_images)
        nature_snyth_imgs = nature_generator(styled_images)

        # Prepare the images to be logged
        input_grid = make_grid(input_images, normalize=True, value_range=(-1, 1)).cpu()
        style_grid = make_grid(styled_images, normalize=True, value_range=(-1, 1)).cpu()
        
        generated_style_grid = make_grid(styled_snyth_imgs, normalize=True, value_range=(-1, 1)).cpu()
        generated_nature_grid = make_grid(nature_snyth_imgs, normalize=True, value_range=(-1, 1)).cpu()

        # Log the images
        logger.writer.add_image('real_natural_images', input_grid, epoch)
        logger.writer.add_image('styled_real_images', style_grid, epoch)

        logger.writer.add_image('nature_generated_images', generated_nature_grid, epoch)
        logger.writer.add_image('styled_generated_images', generated_style_grid, epoch)

        
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_images(engine):
        epoch = engine.state.epoch
        if epoch % LOG_IMAGES_EVERY_N_EPOCH ==0: # for time & disk efficiency  
            log_generated_images(engine, tb_logger, style_generator, nature_generator,epoch)






    
    
