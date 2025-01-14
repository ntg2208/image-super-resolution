from ISR.models import RDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19
from ISR.train import Trainer

lr_train_patch_size = 40
layers_to_extract = [5, 9]
scale = 2
hr_train_patch_size = lr_train_patch_size * scale

rdn  = RDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'x':scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

loss_weights = {
  'generator': 0.0,
  'feature_extractor': 0.0833,
  'discriminator': 0.01
}
losses = {
  'generator': 'mae',
  'feature_extractor': 'mse',
  'discriminator': 'binary_crossentropy'
}

log_dirs = {'logs': './logs', 'weights': './weights/rdn_x2'}

learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

trainer = Trainer(
    generator=rdn,
    discriminator=discr,
    feature_extractor=f_ext,
    lr_train_dir='../../../data/DIV2K/DIV2K_train_LR_bicubic/X2',
    hr_train_dir='../../../data/DIV2K/DIV2K_train_HR',
    lr_valid_dir='../../../data/DIV2K/DIV2K_valid_LR_bicubic/X2',
    hr_valid_dir='../../../data/DIV2K/DIV2K_valid_HR',
    loss_weights=loss_weights,
    learning_rate=learning_rate,
    flatness=flatness,
    dataname='image_dataset',
    log_dirs=log_dirs,
    weights_generator=None,
    weights_discriminator=None,
    n_validation=40,
)

with open("model.txt","w") as f:
  f.write("RDN model: \n")
  rdn.model.summary(print_fn=lambda x: f.write(x + '\n')) #model summary


trainer.train(
    epochs=80,
    steps_per_epoch=500,
    batch_size=16,
    monitored_metrics={}
)

rdn.model.save_weights('../../weight/rdn_x2/ex.hdf5')


