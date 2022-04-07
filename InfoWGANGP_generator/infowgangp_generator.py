# Generating new mitotic cell samples

import os
import numpy as np
import torch
import argparse
from G_model import *
from tqdm import tqdm
import cv2


#%% ---------- Inputing parameters and paths ------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./infowgangp_samples/', help='Path of the output folder for thr generated images')
    parser.add_argument('--models_path', type=str, default=r'.\model', help='Path of the trained model weights')
    parser.add_argument('--generate_num', type=int, default=100, help='Number of mitotic cell images to be generated')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('--z_size', type=int, default=100, help='length of z noise vector')
    parser.add_argument('--dis_category', type=int, default=4, help='number of categories - length of c vector')
    parser.add_argument('--cuda'  , action='store_false', help='enables cuda')
    #opt = parser.parse_args()
    return parser.parse_args()

#%% define some functions

def clear_folder(folder_path):
    """Clear all contents recursively if the folder exists.
    Create the folder if it has been accidently deleted.
    """
    #create_folder(folder_path)
    for the_file in os.listdir(folder_path):
        _file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(_file_path) or os.path.islink(_file_path):
                os.unlink(_file_path)
            elif os.path.isdir(_file_path):
                shutil.rmtree(_file_path)
        except OSError as _e:
            print(_e)
#
def generate_samples(generatedData_path, g_model, gen_num, rand, dis_category, device):

    if generatedData_path is None:
        raise Exception('Please specify path to save data!')
    g_model = g_model.to(device)
    g_model.eval()
    
    for k in tqdm(range(1, gen_num + 1), position=0):
        
        rand_c,label_c = sample_c(batchsize= 1 ,dis_category=dis_category)
        c = torch.randn(1, dis_category).to(device)
        z = torch.randn(1, rand).to(device)
        c.resize_as_(rand_c).copy_(rand_c)
        z.resize_(1, rand, 1, 1).normal_(0, 1)
        c = c.view(-1, dis_category, 1, 1)
        noise = torch.cat([c,z],1)
        new_noise = noise.view(1,rand+dis_category,1,1)

        with torch.no_grad():
            fake = g_model(Variable(new_noise))
        images = fake.data.to('cpu').numpy()
        cv2.imwrite(generatedData_path +'/gan_{0}.png'.format(k), 255*(images.squeeze()+1)/2.) 

#%% Main

def main(opt):
    models_path = opt.models_path
    output_path = opt.output_path
    generate_num = opt.generate_num
    batchsize=opt.batchsize 
    z_size= opt.z_size 
    dis_category= opt.dis_category
    random_seed = 42
    
    
    #%% Making subfolders
    
    if 1- os.path.exists(output_path):
        #output_path = r'./infowgangp_samples/'
        os.makedirs(output_path, exist_ok = True)
    
    print('Output folder_name:', output_path)
    
    # Manual seed
    print('Random seed ={}'.format(random_seed))
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #
    
    # define devices
    print("PyTorch version: {}".format(torch.__version__))
    device = torch.device('cuda' if (torch.cuda.is_available() and opt.cuda) else 'cpu')
    
    if device.type == 'cuda':
        print("CUDA version: {}\n".format(torch.version.cuda))
    elif torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #
    
    
    #%% loading models
    netG = create_model(rand=z_size, dis_category=dis_category)
    netG.load_state_dict(torch.load(os.path.join(models_path, "netG.pth")))
    
    #%% Generating images
    clear_folder(output_path)
    generate_samples(output_path, netG, generate_num, z_size, dis_category, device)

if __name__ == "__main__":
    main(parse_args())



