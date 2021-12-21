import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import albumentations as A
from albumentations_experimental import HorizontalFlipSymmetricKeypoints

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn

from pytorch_lightning.callbacks import ModelCheckpoint

# ----------------------------------------------------------------------------

""" Non-training hyperparameters and machine config """

n_keypoints = 14
val_fraction = 0.8
new_img_size = 128  # images will be resized to this size
verbose = True

KEYPOINT_COLOR = (0, 255, 0)  # color for visualizing

# working device
device = torch.device('cpu')

# configuration for local training without gpu
params = dict()
if device == torch.device('cpu'):
    params['num_workers'] = 0
    params['gpus'] = 0

# configuration for training with gpu
elif device == torch.device('cuda'):
    params['num_workers'] = 2
    params['gpus'] = torch.cuda.device_count()

# ----------------------------------------------------------------------------

""" Visualizing keypoints on image """

# colors for visualizing
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
SALMON = (250, 128, 114)
PURPLE = (128, 0, 128)
DARKPINK = (255, 20, 147)
CHOCOLATE = (210, 105, 30)
INDIGO = (75, 0, 130)
DARKBLUE = (0, 191, 255)
WHITE = (255, 255, 255)

BLACK = (0, 0, 0)

# pairs:
# RED - YELLOW, 0 - 3
# GREEN - BLUE, 1 - 2
# CYAN - DARKPINK, 4 - 9
# MAGENTA - PURPLE, 5 - 8
# ORANGE - SALMON, 6 - 7
# INDIGO - WHITE, 11 - 13

# without pair:
# DARKBLUE - 12
# CHOCOLATE - 10


colors_keypoints = (
    RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, ORANGE,
    SALMON, PURPLE, DARKPINK, CHOCOLATE, INDIGO, DARKBLUE, WHITE
)
EXTENSIONS = ('jpg', 'jpeg', 'bmp', 'png')


# visualizing keypoints
def vis_keypoints(image, keypoints, diameter=3, colors=colors_keypoints):
    image = image.copy()

    num = 0
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, colors[num], -1)
        num += 1

    return image


# R2 loss implementation
def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


# ----------------------------------------------------------------------------

""" Train dataset """


class Keypoints(Dataset):
    def __init__(self,
                 mode,
                 gt_csv,
                 img_dir,
                 fraction: float = val_fraction,
                 transform=None,
                 new_size=new_img_size,
                 num_keypoints=n_keypoints
                 ):

        self._new_size = new_size

        # list of tuples: (img_path, keypoints coordinates np.ndarray of shape (n_keypoints, 2))
        self._items = []

        # transform from albumentations
        self._transform = transform

        filenames = list(gt_csv.keys())
        split_border = int(fraction * len(filenames))

        num = 0
        for filename in filenames:
            if num <= split_border and mode == "val" or num > split_border and mode == "train":
                num += 1
                continue

            keypoints = gt_csv[filename].reshape(num_keypoints, 2)
            if np.any(keypoints < 0):
                continue
            self._items.append((
                os.path.join(img_dir, filename),
                keypoints
            ))
            num += 1

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):

        img_path, keypoints = self._items[index]

        # read image
        image = np.array(Image.open(img_path).convert('RGB')).astype(np.float64)

        # augmentation
        if self._transform:
            transformed = self._transform(image=image, keypoints=keypoints)
            if len(np.array(transformed['keypoints'])) == 14:
                image = transformed['image']
                keypoints = np.array(transformed['keypoints'])

        # resize
        resize_transform = A.Compose([
            A.Resize(height=self._new_size, width=self._new_size, interpolation=cv2.INTER_CUBIC, always_apply=True)
        ], keypoint_params=A.KeypointParams(format='xy', angle_in_degrees=False))

        resized = resize_transform(image=image, keypoints=keypoints)
        image = resized['image']
        keypoints = np.array(resized['keypoints'])

        # normalizing
        for k in range(image.shape[2]):
            mean, std = image[:, :, k].mean(), image[:, :, k].std()
            image[:, :, k] -= mean
            image[:, :, k] /= std

        # ToTensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        keypoints = torch.from_numpy(keypoints.flatten()).float()

        return image, keypoints


# ----------------------------------------------------------------------------

""" Test dataset """


class KeypointsTest(Dataset):
    def __init__(self,
                 img_dir,
                 new_size=new_img_size,
                 ):

        self._new_size = new_size
        self._img_paths = []

        img_filenames = sorted(filter(lambda name: any(name.endswith(ext) for ext in EXTENSIONS), list(os.walk(img_dir))[0][2]))

        for filename in img_filenames:
            self._img_paths.append(
                os.path.join(img_dir, filename)
            )

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, index):

        img_path = self._img_paths[index]

        # read image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image).astype(np.float64)
        shape = image.shape[:2]

        # resize
        resize_transform = A.Compose([
            A.Resize(height=self._new_size, width=self._new_size, interpolation=cv2.INTER_CUBIC, always_apply=True)
        ])

        resized = resize_transform(image=image)
        image = resized['image']

        # normalizing
        for k in range(image.shape[2]):
            mean, std = image[:, :, k].mean(), image[:, :, k].std()
            image[:, :, k] -= mean
            image[:, :, k] /= std

        # ToTensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        shape = torch.from_numpy(np.array(shape))

        return image, img_path, shape


# ----------------------------------------------------------------------------

""" Facepoints detection neural network """


class KeypointsModel(pl.LightningModule):
    def __init__(self, num_keypoints=n_keypoints, verbose=verbose, fast_train=False):
        super().__init__()
        self._verbose = verbose
        self._fast_train = fast_train

        """ Defining possible layers here """

        self.conv1 = nn.Conv2d(3, 64, 3, padding='same')
        self.conv2 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv3 = nn.Conv2d(128, 256, 3, padding='same')
        self.conv4 = nn.Conv2d(256, 512, 3, padding='same')
        self.conv5 = nn.Conv2d(512, 1024, 3, padding='same')

        self.batch1 = nn.BatchNorm2d(64, affine=False)
        self.batch2 = nn.BatchNorm2d(128, affine=False)
        self.batch3 = nn.BatchNorm2d(256, affine=False)
        self.batch4 = nn.BatchNorm2d(512, affine=False)
        self.batch5 = nn.BatchNorm2d(1024, affine=False)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1024 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2 * num_keypoints)

        self.loss = F.mse_loss

    def forward(self, x):
        """ Use for inference only (separate from training_step) """
        x = self.pool(F.relu(self.batch1(self.conv1(x))))
        x = self.pool(F.relu(self.batch2(self.conv2(x))))
        x = self.pool(F.relu(self.batch3(self.conv3(x))))
        x = self.pool(F.relu(self.batch4(self.conv4(x))))
        x = self.pool(F.relu(self.batch5(self.conv5(x))))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        """ The full training loop """
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)
        mae = F.l1_loss(pred, y)
        r2_score = r2_loss(pred, y)

        return {'loss': loss, 'mae': mae.detach(), 'r2_score': r2_score.detach()}

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.094,
            patience=5,
            cooldown=2,
            threshold=0.01,
            verbose=self._verbose
        )

        lr_dict = {
            "scheduler": lr_scheduler,  # REQUIRED: The scheduler instance
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            "monitor": "val_mae"  # metric to to monitor
        }

        if self._fast_train:
            return [optimizer]
        return [optimizer], [lr_dict]

    def predict(self, dataloader):
        predictions = dict()
        for i, data in enumerate(tqdm(dataloader)):
            images, img_paths, true_shapes = data
            with torch.no_grad():
                pred = self(images.to(device))
                for k in range(len(pred)):
                    keypoints = pred[k].cpu().numpy()
                    keypoints = keypoints.reshape(len(keypoints) // 2, 2)
                    true_shape, shape = true_shapes[k].cpu().numpy(), images[k].shape[1:3]
                    scale_x, scale_y = true_shape[0] / shape[0], true_shape[1] / shape[1]
                    keypoints[:, 0] *= float(scale_x)
                    keypoints[:, 1] *= float(scale_y)
                    predictions[img_paths[k].split('/')[-1]] = np.round(keypoints.flatten())
        return predictions

    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """ The full validation loop """
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)
        mae = F.l1_loss(pred, y)
        r2_score = r2_loss(pred, y)

        return {'val_loss': loss, 'val_mae': mae, 'val_r2_score': r2_score}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """ Log and display average train loss and metrics """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['mae'] for x in outputs]).mean()
        avg_r2 = torch.stack([x['r2_score'] for x in outputs]).mean()

        if self._verbose:
            print(f"| Train_mae: {avg_mae:.4f}, Train_r2: {avg_r2:.4f}, Train_loss: {avg_loss:.4f}")

        if not self._fast_train:
            self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
            self.log('train_mae', avg_mae, prog_bar=True, on_epoch=True, on_step=False)
            self.log('train_r2', avg_r2, prog_bar=True, on_epoch=True, on_step=False)

    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """ Log and display average val loss and metrics """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['val_mae'] for x in outputs]).mean()
        avg_r2 = torch.stack([x['val_r2_score'] for x in outputs]).mean()

        if self._verbose:
            print(
                f"[Epoch {self.trainer.current_epoch:3}] Val_mae: {avg_mae:.4f}, Val_r2: {avg_r2:.4f}, "
                f"Val_loss: {avg_loss:.4f}",
                end=" ")

        if not self._fast_train:
            self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
            self.log('val_mae', avg_mae, prog_bar=True, on_epoch=True, on_step=False)
            self.log('val_r2', avg_r2, prog_bar=True, on_epoch=True, on_step=False)


# ----------------------------------------------------------------------------

""" Functions for testing """


def train_detector(gt_csv, img_dir, fast_train=True, verbose=False):
    """ Callbacks and trainer """

    MyModelCheckpoint = ModelCheckpoint(
        dirpath='runs/model_keypoints',
        filename='{epoch}-{val_mae:.4f}',
        monitor='val_mae',
        mode='min',
        save_top_k=1
    )

    max_epochs = 1 if fast_train else 40
    callbacks = [] if fast_train else [MyModelCheckpoint]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=params['gpus'],
        callbacks=callbacks,
        log_every_n_steps=70,
        logger=not fast_train,
        checkpoint_callback=not fast_train
    )

    """ Initializing train and val datasets and dataloaders """

    symm_keypoints = [(0, 3), (1, 2), (4, 9), (5, 8), (6, 7), (11, 13), (10, 10), (12, 12)]
    transform = A.Compose([
        HorizontalFlipSymmetricKeypoints(symm_keypoints),
        A.Rotate(limit=15, interpolation=2, border_mode=cv2.BORDER_CONSTANT)
    ], keypoint_params=A.KeypointParams(format='xy', angle_in_degrees=False))

    ds_train = Keypoints(mode="train", gt_csv=gt_csv, img_dir=img_dir, transform=transform)
    ds_val = Keypoints(mode="val", gt_csv=gt_csv, img_dir=img_dir)

    dl_train = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=params['num_workers'])
    dl_val = DataLoader(ds_val, batch_size=8, shuffle=False, num_workers=params['num_workers'])

    """ Getting model instance and training it """

    model = KeypointsModel(verbose=verbose).to(device)
    trainer.fit(model, dl_train, dl_val)

    return model


def detect(model_filename, test_img_dir):
    model = KeypointsModel.load_from_checkpoint(checkpoint_path=model_filename).to(device)
    ds_test = KeypointsTest(img_dir=test_img_dir)
    dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=params['num_workers'])
    return model.predict(dl_test)
