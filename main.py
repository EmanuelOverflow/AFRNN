#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 10:22:17 2020

@author: emanueldinardo
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import constraints
import numpy as np
import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
import argparse
import os
import glob
import models
from captum.attr import LayerGradCam, LayerGradientXActivation, LayerIntegratedGradients, LayerActivation, NoiseTunnel
import captum.attr._utils.visualization as viz
from captum.robust import FGSM
from captum.robust import PGD
from captum.robust import AttackComparator
import collections
import json
from matplotlib.figure import Figure


def check_weights(module):
    if hasattr(module, 'kernel_constraint'):
        if hasattr(module, 'weight'):
            if module.kernel_constraint is not None:
                print(module.weight.min())
                print(module.weight.max())


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--weights', default='', type=str)
    parser.add_argument('--resume_epoch', default=-1, type=int)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--path', default='', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model', default='fuzzy', type=str)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--tnorm_p', default=1., type=float)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--constraint', default=None, type=str)
    parser.add_argument('--log_every', default=10000, type=int)

    return parser.parse_args()


def train_loop(ds_train, ds_test, model, constraint, criterion, optimizer, epochs, log_dir, init_epoch=0, log_every=0):
    print("====== TRAINING =======")

    train_logdir = os.path.join(log_dir, 'train')
    os.makedirs(train_logdir, exist_ok=True)
    train_writer = SummaryWriter(train_logdir)

    test_logdir = os.path.join(log_dir, 'test')
    os.makedirs(test_logdir, exist_ok=True)
    test_writer = SummaryWriter(test_logdir)

    prev_loss = 9999.
    best_epoch_loss = 9999.
    best_epoch = 0
    epochs_it = tqdm.trange(init_epoch, epochs)
    epochs_it.unpause()
    for epoch in epochs_it:  # epochs
        model.train()
        desc = tqdm.tqdm(ds_train)
        running_loss = 0.0
        correct = 0
        total = 0
        total_steps = len(ds_train)
        for step, (x, y) in enumerate(desc):
            # with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            outputs = model(x.cuda())
            loss = criterion(outputs, y.cuda())
            if step % log_every == 1000 and step != 0:
                train_writer.add_scalar("Loss/step", loss, total_steps * epoch + step)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # wb = [model.fuzzy_layer.weight.min(), model.fuzzy_layer.weight.max()]
            if constraint:
                model.apply(constraint)
            # wa = [model.fuzzy_layer.weight.min(), model.fuzzy_layer.weight.max()]

            # for p in model.parameters():
            #     print(p.name, torch.all(torch.isfinite(p.grad)))

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted.cpu() == y).sum().item()
            accuracy = correct / total
            train_writer.add_scalar("Accuracy/step", accuracy, total_steps * epoch + step)

            desc.set_description(f'[EPOCH {epoch+1}] Loss: {running_loss / (step+1):.4f} - Acc: {100 * accuracy:.4f}')
                                 # f' - WB: {wb[0].data:.5f} {wb[1].data:.5f}'
                                 # f' - WA: {wa[0].data:.5f} {wa[1].data:.5f}')

        train_writer.add_scalar("Loss/epoch", running_loss / total_steps, epoch + 1)
        train_writer.add_scalar("Accuracy/epoch", accuracy, epoch + 1)

        # TEST
        running_loss = 0.0
        correct = 0
        total = 0
        desc = tqdm.tqdm(ds_test)
        model.eval()
        total_steps = len(ds_test)
        with torch.no_grad():
            for step, (x, y) in enumerate(desc):
                outputs = model(x.cuda())
                loss = criterion(outputs, y.cuda())
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted.cpu() == y).sum().item()
                accuracy = correct / total

                desc.set_description(f'[Test] Loss: {running_loss / (step + 1):.4f} - Acc: {100 * accuracy:.4f}')
            else:
                cur_loss = running_loss / total_steps
                test_delta = abs(prev_loss - cur_loss)
                print(f'[Test] Loss: {cur_loss:.4f} - Acc: {100 * correct / total:.4f} '
                      f'- Delta: {test_delta:.4f}')
                test_writer.add_scalar("Loss/delta", test_delta, epoch + 1)

                if cur_loss < best_epoch_loss:
                    best_epoch_loss = cur_loss
                    best_epoch = epoch
                prev_loss = cur_loss

        test_writer.add_scalar("Loss/epoch", running_loss / total_steps, epoch + 1)
        test_writer.add_scalar("Accuracy/epoch", accuracy, epoch + 1)

        save_dict = {
            'model': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'best_epoch': best_epoch
        }

        torch.save(save_dict, os.path.join(log_dir, f"checkpoint_{epoch}.pth"))

    test_writer.add_text("Best metrics", f"Best epoch {best_epoch}, with loss: {best_epoch_loss}", 0)
    train_writer.close()
    test_writer.close()


def explain(model, data, label, path, it):
    gradcam_path = f'{path}/explain/gradcam/input_{it}/'
    ig_path = f'{path}/explain/intgradient/input_{it}'
    igs_path = f'{path}/explain/intgradsmooth/input_{it}'
    ga_path = f'{path}/explain/gradact/input_{it}'
    act_path = f'{path}/explain/act/input_{it}'

    if not os.path.exists(gradcam_path):
        os.makedirs(gradcam_path)
        os.makedirs(ig_path)
        os.makedirs(igs_path)
        os.makedirs(ga_path)
        os.makedirs(act_path)

    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                  **kwargs
                                                  )

        return tensor_attributions

    def save_image_features(path, attr, original_image, algorithm):
        attr = np.transpose(attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        plots, _ = viz.visualize_image_attr_multiple(attr, original_image, fig_size=(18, 6),
                                                     signs=[
                                                         "all",
                                                         "positive",
                                                         "absolute_value",
                                                         "all",
                                                         "positive",
                                                         "absolute_value",
                                                         "all",

                                                     ], use_pyplot=False, show_colorbar=True,
                                                     methods=[
                                                         "original_image",
                                                         "heat_map",
                                                         "heat_map",
                                                         "heat_map",
                                                         "blended_heat_map",
                                                         "blended_heat_map",
                                                         "blended_heat_map",
                                                     ],
                                                     titles=[
                                                         "Original",
                                                         f"{algorithm} Heatmap POS",
                                                         f"{algorithm} Heatmap ABS",
                                                         f"{algorithm} Heatmap ALL",
                                                         f"Overlayed {algorithm} POS",
                                                         f"Overlayed {algorithm} ABS",
                                                         f"Overlayed {algorithm} ALL",
                                                     ])

        plots.savefig(os.path.join(path, f'{name}.png'))

    def save_image_features_channels(attrs, subplot_size):
        plt_fig = Figure(figsize=(12, 12))
        plt_axis_abs = plt_fig.subplots(subplot_size, subplot_size)
        attrs = np.transpose(attrs.cpu().detach().numpy(), (1, 2, 0))
        for channel_n in range(attrs.shape[-1]):
            i = (channel_n // subplot_size)
            j = channel_n % subplot_size
            attrs_plot, _ = viz.visualize_image_attr(grads, None, method="heat_map",
                                                     sign="all", show_colorbar=False,
                                                     plt_fig_axis=(plt_fig, plt_axis[i][j]), fig_size=(8, 8))

    data = data.cuda()
    label = label.cuda()

    original_image = np.transpose((data.cpu().squeeze().detach().numpy()), (1, 2, 0))

    for name, layer in [(n, m) for n, m in model.named_modules()
                        if any(map(n.lower().__contains__, ['fuzzy_layer', 'layer', 'relu']))]:
        gradcam = LayerGradCam(model, layer)
        grads = attribute_image_features(gradcam, data, target=label)  #, baselines=data * 0)
        save_image_features(gradcam_path, grads, original_image, algorithm="GradCAM")

        intgrad = LayerIntegratedGradients(model, layer)
        intgrads = attribute_image_features(intgrad, data, baselines=data * 0, target=label)
        save_image_features(ig_path, intgrads, original_image, algorithm="IG")

        nt = NoiseTunnel(intgrad)
        nt_intgrads = attribute_image_features(nt, data, baselines=data * 0, nt_type='smoothgrad_sq',
                                               nt_samples=100, stdevs=0.2, target=label)
        save_image_features(igs_path, nt_intgrads, original_image,
                            algorithm="IG w SmoothGrad_SQ")

        grad_input = LayerGradientXActivation(model, layer)
        grad_inputs = attribute_image_features(grad_input, data, target=label)
        save_image_features(ga_path, grad_inputs, original_image,
                            algorithm="Grads X Act")

        act_input = LayerActivation(model, layer)
        act_inputs = attribute_image_features(act_input, data)
        save_image_features(act_path, act_inputs, original_image,
                            algorithm="Layer Act")



def attack(model, data, label, path, it):
    ModelResult = collections.namedtuple('ModelResults', 'accuracy score')
    model = model.cpu()

    class ModelResultD(ModelResult):
        def __str__(self):
            return json.dumps({'accuracy': float(self.accuracy), 'score': float(self.score)})

        def __repr__(self):
            return json.dumps({'accuracy': float(self.accuracy), 'score': float(self.score)})

        def to_json(self):
            return {x: float(self.__getattribute__(x)) for x in self._fields}

    def metric(model_out, target):
        if isinstance(target, int):
            target = torch.tensor([target])
        reshaped_target = target.reshape(1, 1)
        score = torch.gather(model_out, 1, reshaped_target).detach()
        _, pred = torch.max(model_out, dim=1)
        acc = (pred == target).float()
        return ModelResultD(accuracy=acc, score=score)

    fgsm = FGSM(model, lower_bound=0, upper_bound=1)

    pgd = PGD(model, lower_bound=0, upper_bound=1)

    comparator = AttackComparator(forward_func=model, metric=metric)
    comparator.add_attack(transforms.RandomRotation(degrees=30), "Random Rotation", num_attempts=100)
    comparator.add_attack(transforms.GaussianBlur(kernel_size=3), "Gaussian Blur", num_attempts=1)
    comparator.add_attack(fgsm, "FGSM", attack_kwargs={"epsilon": 0.15},
                          apply_before_preproc=False, additional_attack_arg_names=["target"], num_attempts=10)
    comparator.add_attack(pgd, "PGD", attack_kwargs={
        "radius": 0.13,
        "step_size": 0.02,
        "step_num": 7,
        "targeted": True
    }, apply_before_preproc=False, additional_attack_arg_names=["target"], num_attempts=10)

    attack_path = f'{path}/attack/'

    if not os.path.exists(attack_path):
        os.makedirs(attack_path)

    data = data
    label = int(label)

    result = comparator.evaluate(data, target=label)
    print(str(result))

    with open(os.path.join(attack_path, f"input_{it}"), "w") as fd:
        fd.write(str(result))


def main(args):
    print(args)

    if args.dataset == "ImageNet":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224)
        ])

        params = {
            'train': {
                'split': 'train',
                'root': '/projects/data/classification/ImageNet2012',
                'transform': transform
            },
            'val': {
                'split': 'val',
                'root': '/projects/data/classification/ImageNet2012',
                'transform': transform
            }
        }

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        params = {
            'train': {
                'root': '/projects/data/classification',
                'train': True,
                'download': True,
                'transform': transform
            },
            'val': {
                'root': '/projects/data/classification',
                'train': False,
                'download': True,
                'transform': transform
            }
        }

    train_data = datasets.__dict__[args.dataset](**params['train'])
    test_data = datasets.__dict__[args.dataset](**params['val'])

    constraint = None
    if args.constraint is not None:
        if args.constraint == 'clip':
            constraint = constraints.ClipWeightValueConstraint()
        elif args.constraint == 'minmax':
            constraint = constraints.MinMaxWeightConstraint()
        elif args.constraint == 'sigmoid':
            constraint = constraints.SigmoidConstraint()
        elif args.constraint == 'gaussian':
            constraint = constraints.GaussianConstraint()
        elif args.constraint == 'sinc2':
            constraint = constraints.Sinc2Constraint()

    model_params = {
        'num_classes': args.num_classes
    }
    if "AFRNN" in args.model:
        model_params.update({
            'tnorm_p': args.tnorm_p,
            'normalized_defuzzy': False,
            'bias': False,
            'constraint': constraint
        })

    model = models.__dict__[args.model](**model_params).cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.NLLLoss()

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    init_epoch = 0

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.weights:
        if args.resume_epoch == -1:
            last_checkpoint = glob.glob(os.path.join(args.weights, f'checkpoint_*.pth'))[-1]
        else:
            last_checkpoint = str(args.resume_epoch)
        weigths = os.path.join(args.weights, last_checkpoint)
        state_dict = torch.load(weigths)
        model.load_state_dict(state_dict['model'])
        model = model.cuda()
        init_epoch = state_dict['epoch']
        optimizer.load_state_dict(state_dict['optimizer'])

        log_dir = args.weights  # os.path.join(*args.weights.split('/')[::-1])
        print(f'Restored from: {args.weights}')
    else:
        log_dir = os.path.join("logs/fit", args.name, datetime.now().strftime("%Y%m%d-%H%M%S"))

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    print("MODEL ARCHITECTURE")
    print(model)

    print(f"Logdir: {log_dir}")
    if args.mode == 'train':
        model.train()
        train_loop(trainloader, testloader, model, constraint, criterion,
                   optimizer, args.epochs, log_dir, init_epoch=init_epoch, log_every=args.log_every)

    # elif args.mode == 'vis':
    #     model.trainable = False
    #     model(x_train[:5])
    #     np.random.seed(10)
    #     test_perm = np.random.permutation(x_test.shape[0])
    #     x_test = x_test[test_perm[:5]]
    #     y_test = y_test[test_perm[:5]]
    #
    #     print(y_test)
    #     for i in range(5):
    #         vis(model, np.expand_dims(x_train[i], 0), f'{args.path}/train', i + 1)
    #         vis(model, np.expand_dims(x_test[i], 0), f'{args.path}/test', i + 1)

    else:
        model.eval()
        test_dataiter = iter(testloader)
        test_images, test_labels = next(test_dataiter)
        test_input = test_images[:10]
        test_label = test_labels[:10]
        test_input.requires_grad = True

        # np.random.seed(10)
        # test_perm = np.random.permutation(x_test.shape[0])

        # x_test = x_test[test_perm[:5]]
        # y_test = y_test[test_perm[:5]]
        _, outputs = torch.max(model(test_input.cuda()), 1)
        op = explain if args.mode == 'explain' else attack
        for i in range(10):
            in_data = test_input[i].unsqueeze(0)
            pred_label = outputs[i]
            out_s = f"P_{classes[pred_label]}_T_{classes[test_label[i]]}_{i}"
            op(model, in_data, test_label[i], f'{args.path}/test', out_s)

    # elif args.mode == 'debug':
    #     model.summary()
    #     model.fit(ds_train, batch_size=args.batch_size, epochs=1, verbose=True, initial_epoch=args.resume_epoch,
    #               validation_data=ds_test)
    #     print(model.layers[0].output)
        # vis(model, np.expand_dims(x_train[0], 0), log_dir)


if __name__ == '__main__':
    args = args_parse()
    main(args)
