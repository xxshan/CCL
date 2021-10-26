import time
import sys

def cycleGAN(opt, data, model, dataset_size, total_iters, epoch_iter, iter_start_time):
    total_iters += opt.gan_batch_size
    epoch_iter += opt.gan_batch_size
    model.set_input(data)         # unpack data from dataset and apply preprocessing

    real_A, real_B, fake_A, rec_A, idt_B = model.optimize_parameters()

    if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
        losses = model.get_current_losses()
        t_comp = (time.time() - iter_start_time) / opt.gan_batch_size

    if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        model.save_networks(save_suffix)

    return real_A, real_B, fake_A, rec_A, idt_B    
