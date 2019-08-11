import argparse
import cl_helper
import json

parser = argparse.ArgumentParser()
parser.add_argument("input", help="input is the image path", type=str)
parser.add_argument("checkpoint", help="checkpoint filename", type=str)
parser.add_argument("--top_k", default=1, help="k most likely classes", type=int)
parser.add_argument("--category_names", default='cat_to_name.json',
                    help="map to real flower names, ie., json file", type=str)
parser.add_argument("--gpu", action="store_true", help="enable gpu for inference")
args = parser.parse_args()

if args.gpu:
    args.gpu = True
else:
    args.gpu = False

print('Loading the checkpoint "{}"....'.format(args.checkpoint))
model, device, architechture, performance_dict, mapping_dict = cl_helper.load_checkpoint(args.checkpoint, gpu=args.gpu)

print('Predicting "{}" using: {}'.format(args.input, device))
probs, classes, names = cl_helper.predict(args.input, args.category_names, model, device, topk=args.top_k)

print('Classes {}'.format(classes))
print('Names {}'.format(names))
print('Probabilities {}'.format(probs))