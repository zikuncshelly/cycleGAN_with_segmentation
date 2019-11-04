import argparse
from PIL import Image
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, help='path to the test images')
    parser.add_argument('--height', type=int, help='imgheight', default=256)
    parser.add_argument('--out_dir', type=str, help='output dir', default='./')

    args = parser.parse_args()
    print(args)
    out_dir = '/scratch/zikunc/cygan_/testGTA/gt'

    imgs = [f for f in os.listdir(args.image_path) if f.endswith('png')]
    for img in imgs:
    	frompath = os.path.join(args.image_path,img)
    	topath = os.path.join(out_dir,img)
    	im = Image.open(frompath)
    	try:
    		im = im.resize((args.height*2,args.height),Image.NEAREST)
    		im.save(topath)
    	except:
    		print(img)