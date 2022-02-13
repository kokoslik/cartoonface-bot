import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm


def train(dataloader, model, criterion, epochs, lr, pools, device, dump_dir='dumps', startepoch=0):
    # Losses & scores
    losses = {'genA': [], 'genB': [], 'disA': [], 'disB': [], 'cycle': [], 'identity': []}
    if startepoch > 0:
        for name, mod in model.items():
            mod.load_state_dict(torch.load(dump_dir+ '/' + name + f'_epoch{startepoch}.pth'))
        with open(dump_dir + '/' + f'loss{startepoch}.pkl', 'rb') as handle:
            losses = pickle.load(handle)

    for key, item in model.items():
        item.train()

    # Create optimizers
    optimizer = {
        "disA": torch.optim.Adam(model["disA"].parameters(),
                                 lr=lr, betas=(0.5, 0.999)),
        "disB": torch.optim.Adam(model["disB"].parameters(),
                                 lr=lr, betas=(0.5, 0.999)),
        "gen": torch.optim.Adam([{'params': model["genA"].parameters()}, {'params': model['genB'].parameters()}],
                                lr=lr, betas=(0.5, 0.999))
    }
    scheduler_lambda = lambda epoch: (1.0 - max(0, epoch - 100) / float(100))
    scheduler = {key: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=scheduler_lambda, verbose=True) for key, opt in
                 optimizer.items()}

    for epoch in tqdm(range(startepoch + 1, epochs)):
        for key, item in model.items():
            item.train()
        # clear_output(wait=True)
        # n_images = 64

        # fixed_latent = torch.randn(n_images, latent_size, 1, 1, device=device)
        # model['generator'].eval()
        # with torch.no_grad():
        #    fake_images = model["generator"](fixed_latent).cpu()
        # model['generator'].train()
        # show_images(fake_images)
        # plt.show()

        for _, lst in losses.items():
            lst.append(0)
        num_batches = 0

        for batchA, batchB in tqdm(zip(dataloader['trainA'], dataloader['trainB'])):
            num_batches += 1
            batchA, batchB = batchA.to(device), batchB.to(device)
            # Train generator
            # Clear generator gradients
            optimizer["gen"].zero_grad()

            fakeA_images = model["genA"](batchB)
            fakeB_images = model["genB"](batchA)
            # fakeA_images = pools['A'].query(model["genA"](batchB))
            # fakeB_images = pools['B'].query(model["genB"](batchA))

            # Reconstructed images
            recA_images = model['genA'](fakeB_images)
            recB_images = model['genB'](fakeA_images)

            cycle_loss = criterion['cycle'](recA_images, batchA) + criterion['cycle'](recB_images, batchB)
            losses['cycle'][-1] += cycle_loss.item()

            idA_images = model["genA"](batchA)
            idB_images = model["genB"](batchB)
            identity_loss = criterion['identity'](idA_images, batchA) + criterion['identity'](idB_images, batchB)
            losses['identity'][-1] += identity_loss.item()

            # Try to fool the discriminator
            predsA = model["disA"](fakeA_images)
            targetsA = torch.ones_like(predsA)
            lossA_g = criterion["gen"](predsA, targetsA)
            losses['genA'][-1] += lossA_g.item()

            predsB = model["disB"](fakeB_images)
            targetsB = torch.ones_like(predsB)
            lossB_g = criterion["gen"](predsB, targetsB)
            losses['genB'][-1] += lossB_g.item()

            loss_g = lossA_g + lossB_g + 10 * cycle_loss + 5.0 * identity_loss

            # Update generator weights
            loss_g.backward()
            optimizer["gen"].step()

            # Train discriminator A
            # Clear discriminator gradients
            optimizer["disA"].zero_grad()

            # Pass real A images through discriminator
            realA_preds = model["disA"](batchA)
            realA_targets = torch.ones_like(realA_preds)
            realA_loss = criterion["disA"](realA_preds, realA_targets)
            # cur_real_score = torch.mean(real_preds).item()

            # Generate fake A images
            fakeA_images = pools['A'].query(fakeA_images)

            # Pass fake images through discriminator
            fakeA_preds = model["disA"](fakeA_images)
            fakeA_targets = torch.zeros_like(fakeA_preds)

            fakeA_loss = criterion["disA"](fakeA_preds, fakeA_targets)
            # cur_fake_score = torch.mean(fake_preds).item()

            # real_score_per_epoch.append(cur_real_score)
            # fake_score_per_epoch.append(cur_fake_score)

            # Update discriminator weights
            loss_disA = 0.5 * (realA_loss + fakeA_loss)
            losses['disA'][-1] += loss_disA.item()
            loss_disA.backward()
            optimizer["disA"].step()
            # loss_d_per_epoch.append(loss_d.item())

            # Train discriminator B
            # Clear discriminator gradients
            optimizer["disB"].zero_grad()

            # Pass real B images through discriminator
            realB_preds = model["disB"](batchB)
            realB_targets = torch.ones_like(realB_preds)
            realB_loss = criterion["disB"](realB_preds, realB_targets)
            # cur_real_score = torch.mean(real_preds).item()

            # Generate fake B images
            fakeB_images = pools['B'].query(fakeB_images)

            # Pass fake images through discriminator
            fakeB_preds = model["disB"](fakeB_images)
            fakeB_targets = torch.zeros_like(fakeB_preds)

            fakeB_loss = criterion["disB"](fakeB_preds, fakeB_targets)

            # Update discriminator weights
            loss_disB = 0.5 * (realB_loss + fakeB_loss)
            losses['disB'][-1] += loss_disB.item()
            loss_disB.backward()
            optimizer["disB"].step()
        for _, lst in losses.items():
            lst[-1] /= num_batches
        for sch in scheduler.values():
            sch.step()
        if (epoch + 1) % 10 == 0:
            for name, mod in model.items():
                torch.save(mod.state_dict(), dump_dir + '/' + name + f'_epoch{epoch}.pth')
            with open(dump_dir + '/' + f'loss{epoch}.pkl', 'wb') as handle:
                pickle.dump(losses, handle)

            batchA = next(iter(dataloader['testA'])).to(device)
            batchB = next(iter(dataloader['testB'])).to(device)
            with torch.no_grad():
                model["genA"].eval()
                model["genB"].eval()
                fakeA_images = model["genA"](batchB)
                fakeB_images = model["genB"](batchA)

                plt.figure(figsize=(8, 10))
                for i in range(5):
                    plt.subplot(5, 4, 4 * i + 1)
                    plt.imshow(np.moveaxis(batchB[i].cpu().detach().numpy(), 0, -1) * 0.5 + 0.5)
                    plt.subplot(5, 4, 4 * i + 2)
                    plt.imshow(np.moveaxis(fakeA_images[i].cpu().numpy(), 0, -1) * 0.5 + 0.5)
                    plt.subplot(5, 4, 4 * i + 3)
                    plt.imshow(np.moveaxis(batchA[i].cpu().detach().numpy(), 0, -1) * 0.5 + 0.5)
                    plt.subplot(5, 4, 4 * i + 4)
                    plt.imshow(np.moveaxis(fakeB_images[i].cpu().numpy(), 0, -1) * 0.5 + 0.5)
                plt.savefig(dump_dir + '/' + f'{epoch}_result.jpg', dpi=300)
            plt.figure(figsize=(12, 9))
            for label, loss in losses.items():
                plt.plot(loss, label=label)
            plt.legend()
            plt.savefig(dump_dir + '/' + f'{epoch}_loss.jpg', dpi=300)
    return losses
