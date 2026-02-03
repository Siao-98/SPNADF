import os
import subprocess
import glob

import logging
from random import choices  # requires Python >= 3.6
import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.use('agg')
import random
from skimage.metrics import structural_similarity as SSIM



def percept_loss(args, gt, output, loss_func):
    gt_channel = gt.repeat(1, 3, 1, 1)  # Repeat channels (B, 1, H, W) -> (B, 3, H, W)
    output_channel = output.repeat(1, 3, 1, 1)

    # Calculate the loss and return the mean
    return torch.mean(loss_func(gt_channel.to('cpu'), output_channel.to('cpu'))).to(args.device)


    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_latest_checkpoint(args):
    if 'disc' in args.model_name:
        gen_path = os.path.join(args.weights_dir, args.model_name + '_' + args.gen_name)
        disc_path = os.path.join(args.weights_dir, args.model_name + '_' + args.disc_name)
        gen_state_dict = sorted(glob.glob(gen_path + '_iter_*.pth'), reverse=True)[0]
        disc_state_dict = sorted(glob.glob(disc_path + '_iter_*.pth'), reverse=True)[0]
        return gen_state_dict, disc_state_dict
    else:
        path = 'add/your/path/here/'  # Example path, replace with actual path if needed
        model_state_dict = path + 'model_state_dict.pth'  
        return model_state_dict


def normalize(data, max_value=255.):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""

    return np.float32(data / max_value)


def mean_normalize(data, axis = 0):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
    mean = np.mean(data, axis=axis)
    return mean, np.float32(data) / np.float32(mean)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def frames_extraction(video_path, frames_no, random_val, dwratio = 2, start_frame=None, downsample=None):
    vid = cv2.VideoCapture(video_path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 15:
        print('*************************')
        print(video_path)
        print('*************************')
    if start_frame == None and total_frames > 15:
        start_frame = random_val
    else:
        start_frame = start_frame
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    seq_list = []
    for _ in range(frames_no):
        _, img = vid.read()
        seq_list.append(img)

    for f in range(len(seq_list)):
        img = seq_list[f]
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        if downsample == True:
            c = 1024
            r = 512
            output_shape = (int(c/dwratio), int(r/dwratio))
        else:
            output_shape = (img.shape[1] + (img.shape[1] % 4), img.shape[0] + (img.shape[0] % 4))
        img = cv2.resize(img, output_shape, interpolation=cv2.INTER_CUBIC)
        seq_list[f] = img

    seq = np.stack(seq_list, axis=0)
    return np.asarray(seq, np.float32)


def compare_psnr(img1, img2, data_range=255.0):
    return 10. * ((data_range ** 2) / ((img1 - img2) ** 2).mean()).log10()


def batch_ssim(img, imclean, data_range=1.0):
    ssim_val = 0
    for b in range(img.shape[0]):
        for i in range(img.shape[1]):
            ssim_val += SSIM(imclean[b, i, 0, :, :].detach().cpu().numpy(), img[b, i, 0, :, :].detach().cpu().numpy(),
                             data_range=data_range)
    #print('ssim: ', ssim_val / (img.shape[0] * img.shape[1]))
    return ssim_val


def batch_psnr(batch, out, gt_r, bin, gt_d, d_out, timestamps, data_range, plotdir, iteration, visualize):
    psnr_r = 0
    psnr_d = 0
    assert out.shape == gt_r.shape
    assert out.shape == bin.shape
    assert d_out.shape == gt_d.shape
    assert d_out.shape == out.shape
    
    #print('out: ', torch.min(out), torch.max(out), out.dtype)

    for b in range(out.shape[0]):
        for i in range(out.shape[1]):
            psnr_r += compare_psnr(gt_r[b, i, :, :, :], out[b, i, :, :, :],
                                   data_range=data_range)
            psnr_d += compare_psnr(gt_d[b, i, :, :, :], d_out[b, i, :, :, :],
                                   data_range=data_range)

    if visualize:
        frame = random.randint(0, out.shape[1] - 1)

        for b in range(out.shape[0]):
            out_seq = out[b, frame, ...].detach().cpu().numpy()
            dout_seq = d_out[b, frame, ...].detach().cpu().numpy()
            bin_seq = bin[b, frame, ...].detach().cpu().numpy()
            ref_seq = gt_r[b, frame, ...].detach().cpu().numpy()
            dep_seq = gt_d[b, frame, ...].detach().cpu().numpy()
            ts_seq = timestamps[b, frame, ...].detach().cpu().numpy()

            out_seq, dout_seq, bin_seq, ref_seq, dep_seq, ts_seq = out_seq.transpose(1, 2, 0) * 255.0, dout_seq.transpose(1, 2, 0) * 255.0, \
                                                               bin_seq.transpose(1, 2, 0) * 255.0, ref_seq.transpose(1, 2, 0) * 255.0, \
                                                                dep_seq.transpose(1, 2, 0) * 255.0, ts_seq.transpose(1, 2, 0) * 255.0
            #print('out_seq: ', np.min(out_seq), np.max(out_seq), out_seq.dtype)

            # out_seq, gt_seq, qis_seq = np.clip(out_seq, 0, 1.0) * 255.0, np.clip(gt_seq, 0, 1.0) * 255.0, np.clip(
            #     qis_seq,
            #     0,
            #     1.0) * 255.0
            out_ref_error = np.abs(out_seq - ref_seq)
            out_dep_error = np.abs(dout_seq - dep_seq)

            out_seq, dout_seq, bin_seq, ref_seq, dep_seq, ts_seq, out_ref_error, out_dep_error = out_seq.astype(
                np.uint8), dout_seq.astype(np.uint8), bin_seq.astype(np.uint8), ref_seq.astype(
                np.uint8), dep_seq.astype(np.uint8), ts_seq.astype(np.uint8), out_ref_error.astype(
                np.uint8), out_dep_error.astype(np.uint8)
            
            #print('out_seq: ', np.min(out_seq), np.max(out_seq), out_seq.dtype, out_seq.shape)

            fig = plt.figure(figsize=(14, 5.5))

            plt.subplot(2, 4, 1)
            plt.imshow(ref_seq[..., 0], cmap='gray')
            plt.axis('off')
            plt.title('GT REF')

            plt.subplot(2, 4, 2)
            plt.imshow(bin_seq[..., 0], cmap='gray')
            plt.axis('off')
            plt.title('INP BIN')

            plt.subplot(2, 4, 3)
            plt.imshow(dep_seq[..., 0], cmap='jet')
            plt.axis('off')
            plt.title('GT DEPTH')

            plt.subplot(2, 4, 4)
            plt.imshow(ts_seq[..., 0], cmap='jet')
            plt.axis('off')
            plt.title('TIMESTAMP')

            plt.subplot(2, 4, 5)
            plt.imshow(out_seq[..., 0], cmap='gray')
            plt.axis('off')
            plt.title('REF RESTORED')

            plt.subplot(2, 4, 6)
            plt.imshow(out_ref_error[..., 0], cmap='gray')
            plt.axis('off')
            plt.colorbar()
            plt.title('mean out_ref_error %0.5f' % (np.mean(out_ref_error)))

            plt.subplot(2, 4, 7)
            plt.imshow(dout_seq[..., 0], cmap='jet')
            plt.axis('off')
            plt.title('DEPTH RESTORED')

            plt.subplot(2, 4, 8)
            plt.imshow(out_dep_error[..., 0], cmap='gray')
            plt.axis('off')
            plt.colorbar()
            plt.title('mean out_dep_error %0.5f' % (np.mean(out_dep_error)))

            #plt.colorbar()
            #plt.show()

            fig.savefig(os.path.join(plotdir, str(iteration) + '_%03d.png' % batch))
            plt.close()

    # print('reflectivity psnr: ', psnr_r / (out.shape[0] * out.shape[1]))
    # print('depth psnr: ', psnr_d / (d_out.shape[0] * d_out.shape[1]))

    return psnr_r / (out.shape[0] * out.shape[1]), psnr_d / (d_out.shape[0] * d_out.shape[1])


def adjust_learning_rate(lr_in, optimizer, epoch, args):
    """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
    lr = lr_in * (args.LR_factor ** (epoch // args.patience))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def get_layer_output_hook(module, input, output):
    print("Layer output shape:", output.shape)
    global layer_output
    layer_output = output

def save_layer_output_as_npy(module, input, output, layer_name="layer", folder="layer_outputs"):


    scale = 2
    os.makedirs(f"{folder}_{scale}", exist_ok=True)

    if isinstance(output, list):
        output = output[int(scale)]  


    output = output.detach().cpu()

    output = torch.squeeze(output)


    batch_size, channels, height, width = output.shape
    output_mat = np.zeros((height, width, 2))

    for i in range(batch_size):
        for j in range(channels):

            channel_output = output[i, j, :, :].numpy()
            output_mat[:,:,j] = channel_output


        mat_path = os.path.join(f"{folder}_{scale}", f"{layer_name}_frame_{i}")
        np.save(mat_path, output_mat)
    print(f"Saved {channels * batch_size} npys from {layer_name}.")

def save_layer_output_as_images(module, input, output, layer_name="layer", folder="layer_outputs"):


    scale = 2
    os.makedirs(f"{folder}_{scale}", exist_ok=True)

    if isinstance(output, list):
        output = output[int(scale)]  


    output = output.detach().cpu()

    output = torch.squeeze(output)
    batch_size, channels, height, width = output.shape
    for i in range(batch_size):
        for j in range(channels):

            channel_output = output[i, j, :, :].numpy()


            min_val = np.min(channel_output)
            max_val = np.max(channel_output)
            norm_output = (channel_output - min_val) / (max_val - min_val)

            norm_output = (norm_output * 255).astype(np.uint8)

            color_img = cv2.applyColorMap(norm_output, cv2.COLORMAP_DEEPGREEN)		

            img_path = os.path.join(f"{folder}_{scale}", f"{layer_name}_frame_{i}_flow_direction_{j}.png")
            cv2.imwrite(img_path, color_img)

    print(f"Saved {channels * batch_size} images from {layer_name}.")


def visualize_optical_flow(flow_x, flow_y, layer_name="layer", folder="optical_flow_outputs", index=0):
    os.makedirs(folder, exist_ok=True)
    

    magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
    

    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # Create an HSV image: Hue corresponds to angle, Saturation is maximum, Value is the magnitude
    hsv = np.zeros((flow_x.shape[0], flow_x.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = angle / 2  # OpenCV uses degrees (0-360), HSV hue is [0-180]
    hsv[..., 1] = 1
    hsv[..., 2] = magnitude
    

    rgb_flow = hsv # cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    

    rgb_flow_uint8 = (rgb_flow * 255).astype(np.uint8)
    

    cv2.imwrite(os.path.join(folder, f"{layer_name}_optical_flow_{index}.png"), rgb_flow_uint8)
    
    print(f"Saved optical flow visualization as {layer_name}_optical_flow_{index}.png")

def save_layer_output_as_optical_flow(module, input, output, layer_name="layer", folder="optical_flow_outputs"):
    os.makedirs(folder, exist_ok=True)

    if isinstance(output, list):
        output = output[0]


    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    else:
        print(f"Unexpected output type: {type(output)}")
        return

    output = np.squeeze(output)


    batch_size, channels, height, width = output.shape
    
    if channels % 2 != 0:
        print("The number of channels should be even, representing optical flow x and y components.")
        return


    for i in range(batch_size):

        for j in range(0, channels, 2):
            flow_x = output[i, j, :, :]
            flow_y = output[i, j + 1, :, :]

            visualize_optical_flow(flow_x, flow_y, layer_name=layer_name, folder=folder, index=i*channels//2 + j//2)
