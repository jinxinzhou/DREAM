import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import shutil

def set_seed(manualSeed=666):
    import random
    import numpy as np
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    os.environ['PYTHONHASHSEED'] = str(manualSeed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val', 'eval'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='Rusume path')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-seed', type=int, default=912)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--order', type=str, default='inf')
    parser.add_argument('--name', type=str, default='sr_ffhq')
    parser.add_argument('--continuous', action='store_true')

    # parse configs
    args = parser.parse_args()
    args.name = "{}_{}".format(args.name, args.order)
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    set_seed(args.seed)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train_log = Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    valid_log = Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    logger.info('order = {}'.format(args.order))

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters(order=args.order)
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_lpips = 0.0
                    avg_const = 0.0
                    avg_fid = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_step)
                    result_hr_path = '{}/hr'.format(result_path)
                    result_sr_path = '{}/sr'.format(result_path)
                    result_lr_path = '{}/lr'.format(result_path)
                    result_fr_path = '{}/fr'.format(result_path)
                    process_path = '{}/process'.format(result_path)
                    os.makedirs(result_path, exist_ok=True)
                    os.makedirs(result_hr_path, exist_ok=True)
                    os.makedirs(result_sr_path, exist_ok=True)
                    os.makedirs(result_lr_path, exist_ok=True)
                    os.makedirs(result_fr_path, exist_ok=True)
                    os.makedirs(process_path, exist_ok=True)



                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    contiuous = args.continuous
                    for _,  val_data in enumerate(val_loader):
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=contiuous)
                        visuals = diffusion.get_current_visuals()
                        for i in range(visuals['HR'].shape[0]):
                            idx += 1
                            hr_img = Metrics.tensor2img(visuals['HR'][i])  # uint8
                            lr_img = Metrics.tensor2img(visuals['LR'][i])  # uint8
                            if contiuous:
                                sr_img_process = (visuals['SR'].permute(1,0,2,3,4))[i] # uint8
                                sample_num = sr_img_process.shape[0]
                                for iter in range(0, sample_num):
                                    sr_img = Metrics.tensor2img(sr_img_process[iter])
                                    os.makedirs('{}/{}/'.format(process_path, str(current_step).zfill(4)), exist_ok=True)
                                    Metrics.save_img(
                                        Metrics.tensor2img(sr_img_process[iter]), '{}/{}/{}_sr_{}.png'.format(process_path, str(current_step).zfill(4), str(iter).zfill(4), str(idx).zfill(3)))
                                sr_img = Metrics.tensor2img(visuals['SR'][-1])  # uint8
                            else:
                                sr_img = Metrics.tensor2img(visuals['SR'][i])

                            # generation
                            Metrics.save_img(
                                hr_img, '{}/{}_{}.png'.format(result_hr_path, current_step, idx))
                            Metrics.save_img(
                                sr_img, '{}/{}_{}.png'.format(result_sr_path, current_step, idx))
                            Metrics.save_img(
                                lr_img, '{}/{}_{}.png'.format(result_lr_path, current_step, idx))
                            avg_psnr += Metrics.calculate_psnr(
                                sr_img, hr_img)
                            avg_ssim += Metrics.calculate_ssim(
                                sr_img, hr_img)
                            avg_lpips += Metrics.calculate_lpips(sr_img, hr_img, device=diffusion.device)
                            avg_const += Metrics.calculate_consistency(sr_img, lr_img)

                            if wandb_logger:
                                wandb_logger.log_image(
                                    f'validation_{idx}', 
                                    np.concatenate((sr_img, hr_img), axis=1)
                                )

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx 
                    avg_lpips = avg_lpips / idx
                    avg_const = avg_const / idx
                    fid = Metrics.calculate_fid(result_hr_path, result_sr_path).item()
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.8f}, ssim: {:.8f}, lpips:  {:.8f}, fid:  {:.8f}, const:  {:.8f}, '.format(
                        current_epoch, current_step, avg_psnr, avg_ssim, avg_lpips, fid, avg_const))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    tb_logger.add_scalar('lpips_vgg', avg_lpips, current_step)
                    tb_logger.add_scalar('const', avg_const, current_step)
                    tb_logger.add_scalar('fid', fid, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_ssim': avg_ssim,
                            'validation/val_lpips': avg_lpips,
                            'validation/val_fid': fid,
                            'validation/val_const': avg_const,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    elif opt['phase'] == 'val':
        logger.info('Begin Model Testing.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        avg_const = 0.0

        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        result_hr_path = '{}/hr'.format(result_path)
        result_sr_path = '{}/sr'.format(result_path)
        result_lr_path = '{}/lr'.format(result_path)
        result_process_path = '{}/process'.format(result_path)
        os.makedirs(result_path, exist_ok=True)
        os.makedirs(result_hr_path, exist_ok=True)
        os.makedirs(result_sr_path, exist_ok=True)
        os.makedirs(result_lr_path, exist_ok=True)
        os.makedirs(result_process_path, exist_ok=True)
        
        contiuous = args.continuous
        for _,  val_data in enumerate(val_loader):
            diffusion.feed_data(val_data)
            diffusion.test(continous=contiuous)
            visuals = diffusion.get_current_visuals()
            for i in range(visuals['HR'].shape[0]):
                idx += 1
                hr_img = Metrics.tensor2img(visuals['HR'][i])  # uint8
                lr_img = Metrics.tensor2img(visuals['LR'][i])  # uint8
                if contiuous:
                    sr_img_process = (visuals['SR'].permute(1,0,2,3,4))[i] # uint8
                    sample_num = sr_img_process.shape[0]
                    for iter in range(0, sample_num):
                        sr_img = Metrics.tensor2img(sr_img_process[iter])
                        os.makedirs('{}/{}/'.format(result_process_path, str(current_step).zfill(4)), exist_ok=True)
                        Metrics.save_img(
                            Metrics.tensor2img(sr_img_process[iter]), '{}/{}/{}_sr_{}.png'.format(process_path, str(current_step).zfill(4), str(iter).zfill(4), str(idx).zfill(3)))
                    sr_img = Metrics.tensor2img(visuals['SR'][-1])  # uint8
                else:
                    sr_img = Metrics.tensor2img(visuals['SR'][i])
        
                Metrics.save_img(
                    hr_img, '{}/hr_{}.png'.format(result_hr_path, str(idx).zfill(3)))
                Metrics.save_img(
                    lr_img, '{}/lr_{}.png'.format(result_lr_path, str(idx).zfill(3)))
                Metrics.save_img(
                    sr_img, '{}/sr_{}.png'.format(result_sr_path, str(idx).zfill(3)))

                # generation
                eval_psnr = Metrics.calculate_psnr(sr_img, hr_img)
                eval_ssim = Metrics.calculate_ssim(sr_img, hr_img)
                eval_lpips = Metrics.calculate_lpips(sr_img, hr_img, device=diffusion.device)
                eval_const = Metrics.calculate_consistency(sr_img, lr_img)

                avg_psnr += eval_psnr
                avg_ssim += eval_ssim
                avg_lpips += eval_lpips
                avg_const += eval_const

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_lpips = avg_lpips / idx
        avg_const = avg_const / idx

        fid = Metrics.calculate_fid(result_sr_path, result_hr_path).item()

        # log
        logger.info('# Validation # PSNR: {:.4f}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4f}'.format(avg_ssim))
        logger.info('# Validation # LPIPS: {:.4f}'.format(avg_lpips))
        logger.info('# Validation # FID: {:.4f}'.format(fid))
        logger.info('# Validation # Consistency: {:.4f}'.format(avg_const))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4f}, ssim：{:.4f}, lpips: {:.4f}, fid: {:.4f}, const：{:.4f}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim, avg_lpips, fid, avg_const))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim),
                'LPIPS': float(avg_lpips),
                'FID': float(fid),
                'Consistency': float(avg_const)
            })
