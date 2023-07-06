import cv2
from models import NoiseNet

from torchvision import transforms
import torch
import argparse
import os
import math
import time
g_main_args = None 
def create_intermediate_frame(bf:cv2.Mat,af:cv2.Mat,comp_num):
    exp = int(min(math.sqrt(comp_num),8))
    cachedir = g_main_args.cache
    expoutdir = os.path.join(g_main_args.cache,"exp_imgs_" + str(int(time.time())))
    if False is os.path.isdir(expoutdir):
        os.makedirs(expoutdir)
    
    bf_imgpath = os.path.join(cachedir,"exp_bf_" + str(int(time.time())) + ".jpg")
    cv2.imwrite(bf_imgpath,bf)
    af_imgpath = os.path.join(cachedir,"exp_af_" + str(int(time.time())) + ".jpg")
    cv2.imwrite(af_imgpath,af)
    cmd = "python ./RIFE/inference_img.py --img \"%s\" \"%s\" --model=./RIFE/train_log --outdir=%s --exp=%d"%(bf_imgpath,af_imgpath,expoutdir,exp)
    os.system(cmd)
    
    dir_files = os.listdir(expoutdir)
    resul_imgs =[]
    step_len = len(dir_files)/comp_num
    for i in range(comp_num):
        cur_index = int(i*step_len)
        img_file = os.path.join(expoutdir,"img%d.png"%(cur_index))
        img = cv2.imread(img_file)
        if(img is not None):
            resul_imgs.append(img)
        else :
            print("[WARN ] read image failed!!")
    for file_name in dir_files:
        try: os.remove(os.path.join(expoutdir,file_name))
        except: pass
    try: os.removedirs(expoutdir)
    except: pass
    try: os.remove(af_imgpath)
    except: pass
    try: os.remove(bf_imgpath)
    except: pass
    return resul_imgs
    

def main():
    
    parser = argparse.ArgumentParser(
                    prog='Video Repair',
                    epilog='Not Help')
    parser.add_argument('--input',help="")
    parser.add_argument('--output',help="")
    parser.add_argument('--batch',default=30,type=int)
    parser.add_argument('--cache',help="",default="./cache")
    parser.add_argument('--model',help="",default="./noise_image.plt")
    args = parser.parse_args()
    global g_main_args
    g_main_args = args
    batchsize = args.batch
    cachedir = args.cache
    inputfile = args.input
    outputfile = args.output
    modelfile = args.model
    print("[INFO ] cache-dir    : %s"%(cachedir))
    print("[INFO ] batch-size   : %d"%(batchsize))
    print("[INFO ] input-file   : %s"%(inputfile))
    print("[INFO ] output-file  : %s"%(outputfile))
    print("[INFO ] model-file   : %s"%(modelfile))
    if(os.path.isdir(cachedir) is False):
        print("[INFO ] Create cache dir %s"%(cachedir))
        os.makedirs(cachedir)
    if(os.path.isfile(inputfile) is False):
        print("[ERROR] The input file does not exist!!")
        return -1
    if(os.path.isfile(modelfile) is False):
        print("[ERROR] The model file does not exist!!")
        return -1
    
    video_capture = cv2.VideoCapture(inputfile)
    if(video_capture is None or video_capture.isOpened() is False):
        print("[ERROR] Unable to open input file. The video may be corrupted.")
        return -2
    
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    print("Video Info: \n FPS:%d \n Size: %dx%d"%(int(video_fps),int(video_width),int(video_height)))
    cur_time = int(time.time())
    pure_video = os.path.join(cachedir,"%d.avi"%(cur_time))
    pure_audio = os.path.join(cachedir,"%d.aac"%(cur_time))
    video_writer = cv2.VideoWriter(pure_video,cv2.VideoWriter_fourcc(*'XVID'),video_fps,(int(video_width),int(video_height)))
    if(video_writer is None or video_writer.isOpened() is False):
        print("[ERROR] Unable to open output file.")
        return -2
    os.system("ffmpeg -loglevel error -i \"%s\" -y \"%s\""%(inputfile,pure_audio))
    tran = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(128,128)),
        transforms.Normalize(mean=(0.456),std=(0.224))
    ])

    model = NoiseNet(in_channels=3)
    model_dict_file = torch.load(modelfile)
    model.load_state_dict(model_dict_file["module"])

    hasvideo = True
    last_goodimg = None
    bad_imgnum = 0
    while hasvideo:
        input_imgs = []
        inputs = []
        for _ in range(batchsize):
            ret,img=video_capture.read()
            if not ret:
                hasvideo = False
                break
            else:
                input_imgs.append(img)
                img = tran(img)
                inputs.append(img.tolist())
        if(len(inputs) == 0): break

        inputs = torch.Tensor(inputs)
        with torch.no_grad():
            tagindexs = torch.max(model(inputs).data, dim=1).indices.tolist()
            for i in range(len(tagindexs)):
                tagindex = tagindexs[i]
                if(tagindex == 1):
                    if(bad_imgnum != 0):
                        # 补全损害的帧
                        inter_imgs = create_intermediate_frame(last_goodimg,input_imgs[i],bad_imgnum)
                        print("[INFO ] inter img num %d"%(len(inter_imgs)))
                        for inter_img in inter_imgs:
                            video_writer.write(inter_img)
                        bad_imgnum = 0
                    last_goodimg = input_imgs[i] # 记录最后一张好的图片
                    video_writer.write(input_imgs[i])
                else:
                    bad_imgnum += 1
    #
    video_writer.release()
    if os.path.exists(pure_audio) :
        os.system("ffmpeg -loglevel error -i \"%s\" -i \"%s\" -y \"%s\""%(pure_video,pure_audio,outputfile))
    else:
        os.system("ffmpeg -loglevel error -i \"%s\" -y \"%s\""%(pure_video,outputfile))
    try: os.remove(pure_audio)
    except: pass
    try: os.remove(pure_video)
    except: pass
    return 0
        
        



if __name__ == '__main__':
    main()
