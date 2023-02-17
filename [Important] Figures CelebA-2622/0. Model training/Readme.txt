1. Data Generation:
	Original Dataset: CelebA
	Sampling strategy: sorted CelebA dataset with descending amount of image samples of each class/identity, then select the top 2622 classes

2. Training Namespace(
	batch_size=32, 
	weight-decay=0.0005, 
	dataset='CelebA-2622', 
	device='0', 
	epochs=300, 
	lr=0.0001, 
	model='vgg16bn', 
	optim='sgd', 
	scheduler='cosa', 
	T_max = max_epoch, 
	seed=2020, 
	suffix='', 
	workers=8
	)

3. Training curves:
	move from laptop Saver [action required]
