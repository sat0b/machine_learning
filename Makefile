
run/gan:
	poetry run python gan.py

run/resnet:
	poetry run python resnet.py

tensorboard:
	tensorboard --logdir=logs/
