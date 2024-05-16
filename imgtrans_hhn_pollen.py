import os
import math
import timeit
import argparse
import numpy as np
import utils
import pickle
import random
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import wandb
from torchvision import datasets, models, transforms
import torch.utils.data as data

os.environ["WANDB_API_KEY"] = "fcff66c814af270f0fa4d6ef837609b0da2cccc4"

parser = argparse.ArgumentParser(description='HHN Project')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='MNIST | FashionMNIST | CIFAR10 | CIFAR100 | SVHN')
parser.add_argument('--datadir', default='datasets', type=str)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='hhnmlp')
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=1024, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--dimensions', default=1, type=int)
parser.add_argument('--transform', default='brightness', type=str)
parser.add_argument('--output', default='.', type=str)
args = parser.parse_args()


img_dir = '/media/khoanam/Data/TUGraz/pytorch-image-classification/img_pollen'
# dataset_folder = 'images_3_types_dry_5050'
# dataset_folder = 'images_3_types_half_hydrated_5050'
# dataset_folder = 'images_3_types_hydrated_5050'
# dataset_folder = 'images_16_types_5050'
# dataset_folder = 'images_5_types_multi_layers_7030'
dataset_folder = 'images_7_types_7030'

args.dataset = dataset_folder


dataset_root_folder = f'{img_dir}/{dataset_folder}'

print(dataset_root_folder)

# Set the train and validation directory path
train_directory = f'{dataset_root_folder}_train'
valid_directory = f'{dataset_root_folder}_val'
test_directory = f'{dataset_root_folder}_test'


# normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
# tr_transform = transforms.Compose([transforms.Resize(32),
#                                    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#                                    transforms.RandomHorizontalFlip(),
#                                    transforms.ToTensor(), normalize])
# val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

# img_size = 224

# Applying transforms to the data
image_transforms_normal_3 = {
    'name': 'image_transforms_normal_3',
    
    'train': transforms.Compose([

        # transforms.Resize(size=img_size + 4),
        # transforms.CenterCrop(size=img_size),
        
        transforms.Resize(100),
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),

        # transforms.ColorJitter(
        #     brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                              [0.1434643,  0.16687445, 0.15344492]),
    ]),
    

    'valid': transforms.Compose([
        # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
        transforms.Resize(100),
        transforms.RandomCrop(96, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                             [0.1434643,  0.16687445, 0.15344492]),
    ]),

  
}

image_transforms = image_transforms_normal_3

# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'test': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),

}


# Create iterators for data loading
dataloaders = {
    'train': data.DataLoader(dataset['train'],
                             # sampler=ImbalancedDatasetSampler(
                             #     dataset['train']),
                             batch_size=args.batchsize),

    'test': data.DataLoader(dataset['test'], batch_size=args.batchsize, shuffle=False,
                             # sampler = ImbalancedDatasetSampler(dataset['train']),
                             num_workers=4, pin_memory=True, drop_last=False),
    

}


 
def main():
    start = timeit.default_timer()

    ######## shape parameters
    nchannels, nclasses = 3, len(dataset['train'].classes)
    # if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST': nchannels = 1
    # if args.dataset == 'CIFAR100': nclasses = 100
    # if args.dataset == 'imagenet': nclasses = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######## download datasets
    # kwargs = {'num_workers': 4, 'pin_memory': True}
    # train_dataset = utils.load_data("train", args.dataset, args.datadir)
    # train_dataset = dataset['train']
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
    train_loader = dataloaders['test']
    
    # test_dataset = utils.load_data("test", args.dataset, args.datadir)
    # test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)

    test_loader = dataloaders['test']
    
    # for images, labels in train_loader:
    #     print(images.shape, labels.shape)  # prints [batch_size, 1, 28, 28] and [batch_size]
    #     break  # remove break to go through all batches
        
     
    ######## prepare model structure
    model, save_dir = utils.prepare_model(args, nchannels, nclasses, hin=1)
    wandb.init(project=f"SCN_imgtrans_{args.dataset}", 
                entity="caonam", 
                name=f"SCN_{args.transform}_{save_dir}")
    

    
    model.to(device)
    print(model)
    print(utils.count_model_parameters(model))

    ######## train model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def train(dataloader, model, loss_fn, optimizer):
        for batch, (X, y) in enumerate(tqdm(dataloader, desc='Training')):
            param = random.uniform(0.2, 2.0)
            X, y = X.to(device), y.to(device)
            if args.transform == "brightness":
                X = TF.adjust_brightness(X, brightness_factor=param)
            elif args.transform == "contrast":
                X = TF.adjust_contrast(X, contrast_factor=param)
            elif args.transform == "saturation":
                X = TF.adjust_saturation(X, saturation_factor=param)
            elif args.transform == "sharpness":
                X = TF.adjust_sharpness(X, sharpness_factor=param)
            Hyper_X = Tensor([param]).to(device)

            pred = model(X, Hyper_X)
            loss = loss_fn(pred, y)
            
            beta1 = model.hyper_stack(Hyper_X)
            param2 = random.uniform(0.2, 2.0)
            beta2 = model.hyper_stack(Tensor([param2]).to(device))
            loss += pow(cos(beta1, beta2),2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    def validate(dataloader, model, loss_fn):
        param = random.uniform(0.2, 2.0)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                if args.transform == "brightness":
                    X = TF.adjust_brightness(X, brightness_factor=param)
                elif args.transform == "contrast":
                    X = TF.adjust_contrast(X, contrast_factor=param)
                elif args.transform == "saturation":
                    X = TF.adjust_saturation(X, saturation_factor=param)
                elif args.transform == "sharpness":
                    X = TF.adjust_sharpness(X, sharpness_factor=param)
                Hyper_X = Tensor([param]).to(device)

                pred = model(X, Hyper_X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
        print(f"Test with param={param}: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct, test_loss

    for t in range(args.epochs):
        print(f"=================\n Epoch: {t + 1} \n=================")
        train(train_loader, model, loss_fn, optimizer)
        test_acc, test_loss = validate(test_loader, model, loss_fn)
        wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    print("Done!")
    
    wandb.finish()

    ######## test model
    def test(dataloader, model, loss_fn, param):
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                if args.transform == "brightness":
                    X = TF.adjust_brightness(X, brightness_factor=param)
                elif args.transform == "contrast":
                    X = TF.adjust_contrast(X, contrast_factor=param)
                elif args.transform == "saturation":
                    X = TF.adjust_saturation(X, saturation_factor=param)
                elif args.transform == "sharpness":
                    X = TF.adjust_sharpness(X, sharpness_factor=param)
                Hyper_X = Tensor([param]).to(device)

                pred = model(X, Hyper_X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
        print(f"Test with param={param}: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct

    acc = []
    for param in tqdm(np.arange(0.2, 2, 0.05), desc='Testing'):
        acc.append(test(test_loader, model, loss_fn, param))

    ######## test model fixed degree
    def test_fixed(dataloader, model, loss_fn, param):
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                if args.transform == "brightness":
                    X = TF.adjust_brightness(X, brightness_factor=param)
                elif args.transform == "contrast":
                    X = TF.adjust_contrast(X, contrast_factor=param)
                elif args.transform == "saturation":
                    X = TF.adjust_saturation(X, saturation_factor=param)
                elif args.transform == "sharpness":
                    X = TF.adjust_sharpness(X, sharpness_factor=param)
                Hyper_X = Tensor([1.0]).to(device) # fixed

                pred = model(X, Hyper_X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /=  len(dataloader)
        correct /= len(dataloader.dataset)
        print(f"Test with param={param}: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct

    acc_fixed = []
    for param in tqdm(np.arange(0.2, 2, 0.05), desc='Testing'):
        acc_fixed.append(test_fixed(test_loader, model, loss_fn, param))

    ######## compute beta space
    beta_space = []
    for param in np.arange(0.2, 2, 0.05):
        Hyper_X = Tensor([param]).to(device)
        beta_space.append(model.hyper_stack(Hyper_X).cpu().detach().numpy())

    beta_space = np.stack(beta_space)
    print(beta_space.shape)

    hhn_dict = {'acc': acc, 'acc_fixed': acc_fixed, 'beta_space': np.array(beta_space)}

    ######## write to the bucket
    destination_name = f'{args.output}/{args.transform}/HHN/{save_dir}'
    os.makedirs(destination_name, exist_ok=True)
    np.save(f'{destination_name}/acc.npy', pickle.dumps(hhn_dict))

    torch.save(model.state_dict(), f'{destination_name}/model.pt')

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()
