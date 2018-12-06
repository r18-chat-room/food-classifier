#!/usr/bin/bash

python ./retrain.py --image_dir ./photos/ --saved_model_dir ./model/ --bottleneck_dir ./bottleneck/ --how_many_training_steps 20 --output_labels ./output/output_labels.txt --output_graph ./output/retrain.pb --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1


