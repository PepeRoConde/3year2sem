import argparse
import os
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn

from ship_classifier import ShipClassifier
from ship_dataset import ShipDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Ship Classification Training Script')
    
    # Dataset parameters
    parser.add_argument('--root_dir', type=str, default='/Users/pepe/carrera/3/2/vca/practicas/p2',
                        help='Root directory containing the image folders')
    parser.add_argument('--data_augmentation', action='store_true', 
                        help='Apply data augmentation and include cropped ship images')
    parser.add_argument('--docked', action='store_true',
                        help='Include docked status in labels')
    parser.add_argument('--not_train', action='store_true',
                        help='Not train the model, used in company of --test_images')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data to total data')
    
    # Model parameters
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights for the model')
    parser.add_argument('--model_path', type=str, default='modelParams',
                        help='Path to save or load the model')
    parser.add_argument('--load_model', action='store_true',
                        help='Load a pretrained model instead of training')
    parser.add_argument('--unbalanced', action='store_true',
                        help='Unless specified, class balancing will be applied')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--lr_patience', type=int, default=2,
                        help='Patience for reducing learning_rate')
    parser.add_argument('--learning_rate', type=float, default=4,
                        help='Learning rate for optimizer')
    parser.add_argument('--l2_lambda', type=float, default=0.0,
                        help='Lambda for L2 weight decay regularization.')
    
    # Testing parameters
    parser.add_argument('--test_images', nargs='+', default=[],
                        help='List of image paths to test individually')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    '''
    # Basic run with data augmentation and pretrained model
    python ship_classifier.py --data_augmentation --pretrained
    
    # Change learning rate and batch size
    python ship_classifier.py --learning_rate 0.0005 --batch_size 256
    
    # Run with docked classification and more epochs
    python ship_classifier.py --docked --num_epochs 20 --patience 5
    
    # Load a previously trained model and test it
    python ship_classifier.py --load_model --model_path my_saved_model
    
    # Test specific images with a trained model
    python ship_classifier.py --load_model --test_images imagen2.jpg imagen3.jpg
    '''
    
    # Print the configuration
    print("\nRunning with the following configuration:")
    print(f"Root directory: {args.root_dir}")
    print(f"Data Augmentation: {args.data_augmentation}")
    print(f"Docked classification: {args.docked}")
    print(f"Not training model: {args.not_train}")
    print(f"Training data ratio: {args.train_ratio}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Model path: {args.model_path}")
    print(f"Loading pretrained model: {args.load_model}")
    print(f"Class balancing: {'Unbalanced' if args.unbalanced else 'Balanced'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"L2 regularization lambda: {args.l2_lambda}")
    print(f"Learning rate decay patience: {args.lr_patience}")
    print(f"Test images: {args.test_images}\n")

    
    # Set up datasets
    trainset = ShipDataset(
        root_dir=args.root_dir, 
        train=True, 
        dataAugmentation=args.data_augmentation,
        docked=args.docked,
        train_ratio=args.train_ratio
    )
    
    testset = ShipDataset(
        root_dir=args.root_dir, 
        train=False, 
        dataAugmentation=False,  # No augmentation for test set
        docked=args.docked,
        train_ratio=args.train_ratio
    )



    if not args.unbalanced:
        class_counts = Counter(trainset.labels)
        print(f'\n--\nClass counts before balancing: {class_counts}')
        weights = [1.0 / class_counts[label] for label in trainset.labels]
        print(f'weights for balancing: {Counter(weights)}\n--\n')
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle=False
    else:
        sampler=None
        shuffle=True
    
    # Set up data loaders
    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=shuffle,
        sampler=sampler
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Create classifier
    classifier = ShipClassifier(pretrained=args.pretrained, 
                                docked=args.docked)
    
    if args.load_model:
        if args.docked:
            # Load a pre-trained model
            try:
                try:
                    classifier.partial_load_model(args.model_path)
                except:
                    classifier.load_model(args.model_path)
            except:
                print(f'Model not loaded, probably due to an error with accesing {args.model_path}.')
            # Load a pre-trained model
        else:
            try:
                classifier.load_model(args.model_path)
            except:
                print(f'Model not loaded, probably due to an error with accesing {args.model_path}.')

    
        

    
        
    if not args.not_train:
        # Train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            [param for param in classifier.model.parameters() if param.requires_grad], 
            lr=args.learning_rate
        )
        
        model, history = classifier.train_model(
            train_loader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=args.num_epochs,
            patience=args.patience,
            lr_patience=args.lr_patience,
            l2_lambda=args.l2_lambda
        )
        
        # Test and plot results
        test_acc, test_accuracies, f1, cm = classifier.test_model(testloader)
        classifier.plot_metrics(
            history, 
            test_acc, 
            dataAugmentation=args.data_augmentation, 
            pretrained=args.pretrained,
            cm=cm)
        
        classifier.plotgrid(testset)
        
        # Save the model
        classifier.save_model(args.model_path)
    
    # Test individual images if provided
    if args.test_images:
        classifier.load_model(args.model_path)
        print("\nTesting individual images:")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        test_single_images(classifier, args.test_images, device=device,docked=args.docked)
