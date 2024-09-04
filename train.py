from lib.model import get_model_optimizer,get_losses
from lib.utils import attach_ignite,attach_checkpoint_handler
from config.config import Config
import torch
from ignite.engine import Engine
from lib.data import train_dl

torch.autograd.set_detect_anomaly(True)


def cycle_gan_training():

    style_generator,style_disc,style_gen_optimizer,style_disc_optimizer =\
          get_model_optimizer()
    nature_generator,nature_disc,nature_gen_optimizer,nature_disc_optimizer =\
          get_model_optimizer()

    l1_loss_fn , mse_loss_fn = get_losses()

    def discriminator_loss(disc,fake_imgs,real_imgs,loss_fn):
        real_preds = disc(real_imgs)
        fake_preds = disc(fake_imgs)

        real_loss = loss_fn(real_preds,torch.ones_like(real_preds))
        fake_loss = loss_fn(fake_preds,torch.zeros_like(real_preds))

        return (real_loss+fake_loss)/2

    
    def train_step(engine,batch):
        ''' Train steps:
        
        ### -1- Adversarial train ###
        # 1-1- Generate fake images from both generators
        # 1-2- update discriminators to distinguish real / fake images
        # 1-3- update generators to generate real-like images in same style
        
        ### -2- Cycle consistency train ###
        # 2-1- update generators to verify cycle consistency property

        '''
        normal_img_t,style_img_t = batch

        ### -1- Adversarial train ###

        # FIRST : Update discriminators
        style_generator.eval()
        nature_generator.eval()
        style_disc.train()
        nature_disc.train()
        style_disc.zero_grad()
        nature_disc.zero_grad()
        with torch.no_grad():
            styled_snyth_imgs = style_generator(normal_img_t)
            nature_snyth_imgs = nature_generator(style_img_t)
        
        style_disc_loss = discriminator_loss(style_disc,styled_snyth_imgs,style_img_t,
                                             mse_loss_fn)
        
        nature_disc_loss = discriminator_loss(nature_disc,nature_snyth_imgs,normal_img_t,
                                             mse_loss_fn)
        

        style_disc_loss.backward()
        style_disc_optimizer.step()
        nature_disc_loss.backward()
        nature_disc_optimizer.step()

        # SECOND : Update generators

        ### 2-1 Adversarial loss
        style_generator.train()
        nature_generator.train()
        style_disc.eval()
        nature_disc.eval()
        style_generator.zero_grad()
        nature_generator.zero_grad()

        styled_snyth_imgs = style_generator(normal_img_t)
        nature_snyth_imgs = nature_generator(style_img_t)

        styled_snyth_preds = style_disc(styled_snyth_imgs)
        nature_snyth_preds = nature_disc(nature_snyth_imgs)

        adv_styled_gen_loss = mse_loss_fn(styled_snyth_preds,torch.ones_like(styled_snyth_preds))
        adv_nature_gen_loss = mse_loss_fn(nature_snyth_preds,torch.ones_like(nature_snyth_preds))

        ### 2-2 Cycle consistency loss 

        styled_inverse_images = nature_generator(styled_snyth_imgs)
        nature_inverse_images = style_generator(nature_snyth_imgs)

        nature_cycle_loss = l1_loss_fn(normal_img_t,styled_inverse_images)
        style_cycle_loss = l1_loss_fn(style_img_t,nature_inverse_images)

        cycle_loss = (nature_cycle_loss+style_cycle_loss)*Config.lambda_cycle

        gen_loss = cycle_loss+adv_styled_gen_loss+adv_nature_gen_loss

        gen_loss.backward()
        style_gen_optimizer.step()
        nature_gen_optimizer.step()

        return {
        'style_disc_loss': style_disc_loss.item(),
        'nature_disc_loss': nature_disc_loss.item(),
        'adv_styled_gen_loss': adv_styled_gen_loss.item(),
        'adv_nature_gen_loss': adv_nature_gen_loss.item(),
        'cycle_loss': (nature_cycle_loss+style_cycle_loss).item()
    }

    trainer = Engine(train_step)

    models = {
        'style_generator': style_generator,
        'style_disc': style_disc,
        'nature_generator': nature_generator,
        'nature_disc': nature_disc
    }

    optimizers = {
        'style_gen_optimizer': style_gen_optimizer,
        'style_disc_optimizer': style_disc_optimizer,
        'nature_gen_optimizer': nature_gen_optimizer,
        'nature_disc_optimizer': nature_disc_optimizer
    }

    attach_checkpoint_handler(trainer, models, optimizers)

    attach_ignite(trainer,style_generator,nature_generator)

    trainer.run(train_dl,max_epochs=Config.num_epochs)



if __name__=="__main__":
    cycle_gan_training()