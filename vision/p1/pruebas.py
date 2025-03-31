from torch.utils.data import DataLoader
from ship_classifier import ShipClassifier
from ship_dataset import ShipDataset

dataAugmentation = True
docked=True
pretrained=False

classifier = ShipClassifier(pretrained=pretrained,docked=docked)
classifier.load_model('modelParams')
classifier.model.to('mps')

trainset = ShipDataset(root_dir='/Users/pepe/carrera/3/2/vca/practicas/p2', 
                       train=True, 
                       dataAugmentation=dataAugmentation, 
                       docked=docked,
                       train_ratio=0.8)
testset = ShipDataset(root_dir='/Users/pepe/carrera/3/2/vca/practicas/p2', 
                      train=False, 
                      dataAugmentation=False, 
                      docked=docked,
                      train_ratio=0.8)

testloader = DataLoader(
        trainset, 
        batch_size=512, 
        num_workers=8)


testloader = DataLoader(
        testset, 
        batch_size=512, 
        num_workers=8)

classifier.plotgrid(testset,rows=8,cols=8)
