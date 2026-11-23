import os
import numpy as np
import torch
from nets.hrnet_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics_multi import f_score
from utils.training_checkpoint import atomic_torch_save

def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0, dwa_history=None, best_val_loss=None):
    total_loss      = 0
    total_loss_e      = 0
    total_loss_d      = 0
    total_f_score   = 0

    val_loss        = 0
    val_loss_e        = 0
    val_loss_d        = 0
    val_f_score     = 0
    w1              = 0
    w2              = 0
    a1              = 1.0
    a2              = 1.0
    dd              = 1.0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        if epoch + 1 <= 2:
            w1 = 1
            w2 = 1
        else:
            history_e = dwa_history["losses_e"] if dwa_history is not None else loss_history.call_e()
            history_d = dwa_history["losses_d"] if dwa_history is not None else loss_history.call_d()
            w1 = history_e[-1] / history_e[-2]
            w2 = history_d[-1] / history_d[-2]

        imgs, pngs_e, pngs_d, labels_e, labels_d = batch
        with torch.no_grad():
            cls_weights_1 = np.array(cls_weights[0], np.float32)
            cls_weights_2 = np.array(cls_weights[1], np.float32)
            weights_1 = torch.from_numpy(cls_weights_1)
            weights_2 = torch.from_numpy(cls_weights_2)

            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs_e  = pngs_e.cuda(local_rank) 
                pngs_d  = pngs_d.cuda(local_rank)              
                labels_e = labels_e.cuda(local_rank)
                labels_d = labels_d.cuda(local_rank)
                weights_1 = weights_1.cuda(local_rank)
                weights_2 = weights_2.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            if focal_loss:
                loss_e = Focal_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                loss_d = Focal_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])
            else:
                loss_e = CE_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                loss_d = CE_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])

            if dice_loss:
                main_dice_e = Dice_loss(outputs[0], labels_e)
                main_dice_d = Dice_loss(outputs[1], labels_d)
                loss_e      = loss_e + main_dice_e
                loss_d      = loss_d + main_dice_d
                
            with torch.no_grad():
                _f_score_e = f_score(outputs[0], labels_e)
                _f_score_d = f_score(outputs[1], labels_d)

            loss = loss_d + loss_e
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)

                if focal_loss:
                    loss_e = Focal_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                    loss_d = Focal_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])
                else:
                    loss_e = CE_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                    loss_d = CE_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])

                if dice_loss:
                    main_dice_e = Dice_loss(outputs[0], labels_e)
                    main_dice_d = Dice_loss(outputs[1], labels_d)
                    loss_e      = loss_e + main_dice_e
                    loss_d      = loss_d + main_dice_d

                with torch.no_grad():
                    _f_score_e = f_score(outputs[0], labels_e)
                    _f_score_d = f_score(outputs[1], labels_d)
            sum = np.exp(w1 / 2.0) + np.exp(w2 / 2.0)
            a1 = 2*np.exp(w1 / 2.0) / sum
            a2 = 2*np.exp(w2 / 2.0) / sum

            loss_1 = loss_e
            loss_2 = dd * loss_d
            loss = a1 * loss_1 + a2 * loss_2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += a1 * loss_e.item()
        total_loss_e += a1 * loss_e.item()
        total_loss += dd*a2 * loss_d.item()
        total_loss_d += dd*a2 * loss_d.item()
        total_f_score += _f_score_e.item()
        total_f_score += _f_score_d.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    val_sample_count = 0
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs_e, pngs_d, labels_e, labels_d = batch
        batch_samples = imgs.shape[0]
        with torch.no_grad():
            cls_weights_1 = np.array(cls_weights[0], np.float32)
            cls_weights_2 = np.array(cls_weights[1], np.float32)
            weights_1 = torch.from_numpy(cls_weights_1)
            weights_2 = torch.from_numpy(cls_weights_2)

            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs_e  = pngs_e.cuda(local_rank) 
                pngs_d  = pngs_d.cuda(local_rank)              
                labels_e = labels_e.cuda(local_rank)
                labels_d = labels_d.cuda(local_rank)
                weights_1 = weights_1.cuda(local_rank)
                weights_2 = weights_2.cuda(local_rank)                

            outputs     = model_train(imgs)

            if focal_loss:
                loss_e = Focal_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                loss_d = Focal_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])
            else:
                loss_e = CE_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                loss_d = CE_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])

            if dice_loss:
                main_dice_e = Dice_loss(outputs[0], labels_e)
                main_dice_d = Dice_loss(outputs[1], labels_d)
                loss_e      = loss_e + main_dice_e
                loss_d      = loss_d + main_dice_d
            _f_score_e = f_score(outputs[0], labels_e)
            _f_score_d = f_score(outputs[1], labels_d)

            val_loss += (dd*a2 * loss_d.item() + a1 * loss_e.item()) * batch_samples
            val_loss_e += a1 * loss_e.item() * batch_samples
            val_loss_d += dd*a2 * loss_d.item() * batch_samples
            val_f_score += (_f_score_e.item() + _f_score_d.item()) * batch_samples
            val_sample_count += batch_samples
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / val_sample_count,
                                'f_score'   : val_f_score / val_sample_count,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        train_loss = total_loss / epoch_step
        train_loss_e = total_loss_e / epoch_step
        train_loss_d = total_loss_d / epoch_step
        validation_loss = val_loss / val_sample_count
        validation_loss_e = val_loss_e / val_sample_count
        validation_loss_d = val_loss_d / val_sample_count
        loss_history.append_loss(epoch + 1, train_loss, train_loss_e, train_loss_d, validation_loss, validation_loss_e, validation_loss_d)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss, validation_loss))
        print('Element Loss: %.3f || Element Val Loss: %.3f || Defect Loss: %.3f || Defect Val Loss: %.3f ' % (train_loss_e, validation_loss_e, train_loss_d, validation_loss_d))
        print('w1: %.3f || w2: %.3f ' % (w1, w2))
        print('a1: %.3f || a2: %.3f ' % (a1, a2))
        with open(os.path.join(save_dir, "weights.txt"), 'a') as f:
            f.write(str(a1))
            f.write(", ")
            f.write(str(a2))
            f.write("\n")
            
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            atomic_torch_save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), train_loss, validation_loss)))

        use_resumable_best = dwa_history is not None
        should_save_best = (best_val_loss is None or validation_loss <= best_val_loss) if use_resumable_best else (len(loss_history.val_loss) <= 1 or validation_loss <= min(loss_history.val_loss))
        if should_save_best:
            print('Save best model to best_epoch_weights.pth')
            atomic_torch_save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            best_val_loss = validation_loss
            
        atomic_torch_save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        if dwa_history is not None:
            dwa_history["losses_e"].append(train_loss_e)
            dwa_history["losses_d"].append(train_loss_d)
        epoch_metrics = {
            "train_loss": train_loss,
            "train_loss_element": train_loss_e,
            "train_loss_defect": train_loss_d,
            "val_loss": validation_loss,
            "val_loss_element": validation_loss_e,
            "val_loss_defect": validation_loss_d,
            "dwa_a1": a1,
            "dwa_a2": a2,
        }
        return dwa_history, best_val_loss, epoch_metrics
