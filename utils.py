import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt
import torch

def training_curve_plot(title, train_costs, test_costs, train_accuracy, test_accuracy,
                        batch_size, learning_rate, num_epochs, elapsed):

    plt.style.use('seaborn-v0_8-darkgrid')

    lg, md, sm = 15, 12, 10


    fig, axs = plt.subplots(1, 2, figsize=(12, 6))


    # Subtitle (cleaner formatting)
    elapsed_min, elapsed_sec = divmod(elapsed, 60)
    sub = (f'Batch size: {batch_size}   |   LR: {learning_rate}   |   '
           f'Epochs: {num_epochs}   |   Time: {int(elapsed_min)}m {elapsed_sec:.0f}s')

    fig.subplots_adjust(top=0.82)

    fig.suptitle(title, fontsize=lg, weight='bold')
    fig.text(0.5, 0.91,sub, ha='center', fontsize=md, alpha=0.8)

    #plt.tight_layout(rect=[0, 0, 1, 0.85])  # <-- THIS fixes it

    x = np.linspace(1, num_epochs, len(train_costs))
    train_color = 'darkorchid'
    test_color = 'limegreen'

    # ---- COST PLOT ----
    axs[0].plot(x, train_costs, color=train_color, linewidth=2, label='Train')
    axs[0].plot(x, test_costs, color=test_color, linewidth=2, label='Test')

    axs[0].scatter(x[-1], train_costs[-1], color=train_color, s=60)
    axs[0].scatter(x[-1], test_costs[-1], color=test_color, s=60)
    axs[0].annotate(f"{train_costs[-1]:.3f}",
                (x[-1], train_costs[-1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=10)

    axs[0].annotate(f"{test_costs[-1]:.3f}",
                    (x[-1], test_costs[-1]),
                    textcoords="offset points",
                    xytext=(5, -10),
                    fontsize=10)

    axs[0].set_title('Loss', fontsize=md)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Cost')
    axs[0].legend()

    # ---- ACCURACY PLOT ----
    axs[1].plot(x, train_accuracy, color=train_color, linewidth=2, label='Train')
    axs[1].plot(x, test_accuracy, color=test_color, linewidth=2, label='Test')

    axs[1].scatter(x[-1], train_accuracy[-1], color=train_color, s=60)
    axs[1].scatter(x[-1], test_accuracy[-1], color=test_color, s=60)
    # Train point label
    axs[1].annotate(f"{train_accuracy[-1]:.3f}",
                    (x[-1], train_accuracy[-1]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=10)

    # Test point label
    axs[1].annotate(f"{test_accuracy[-1]:.3f}",
                    (x[-1], test_accuracy[-1]),
                    textcoords="offset points",
                    xytext=(5, -10),
                    fontsize=10)

    axs[1].set_title('Dice Coefficient', fontsize=md)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Dice Coefficient')

    axs[1].legend()

    axs[0].set_ylim(0, max(max(train_costs), max(test_costs)) * 1.1)
    axs[1].set_ylim(0, 1)  # Dice is usually between 0 and 1

    plt.show()
    plt.style.use('default')


def to_rgb(img_tensor):
    img = img_tensor.squeeze().cpu().numpy()  # (2, H, W)

    r, g = img[0], img[1]
    b = np.zeros_like(r)

    rgb = np.stack([r, g, b], axis=-1)

    # Normalize for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    return rgb


def best_worst(how_many, dataset, model, device, thresh=0.6):
    model.eval()

    samples = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            X = sample["image"].unsqueeze(0).to(device) #.unsqueeze(0)
            Y = sample["label"].unsqueeze(0).to(device)

            G = model(X).squeeze(0) #torch.sigmoid(model(X)).squeeze(0)
            dice = dice_coef(G, Y, "inference").item()

            pred = (torch.sigmoid(G) > thresh).float()


            samples.append({
                "idx": i,
                "dice": dice,
                "img": sample["image"].squeeze(0),   # keep CPU version
                "gt": sample["label"],
                "pred": pred.squeeze(0)
            })

    # Sort
    samples_sorted = sorted(samples, key=lambda x: x["dice"])

    worst = samples_sorted[:how_many]
    best = samples_sorted[-how_many:][::-1]

    return best, worst


def plot_four_samples(samples):
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))

    for row, s in enumerate(samples):
        idx = s["idx"]
        dice = s["dice"]
        img = s["img"]
        gt = s["gt"].squeeze()
        pred = s["pred"].cpu()

        # Input
        axes[row, 0].imshow(to_rgb(img))
        axes[row, 0].set_title(f"Sample ID: {idx+1}")
        axes[row, 0].axis('off')

        # GT
        axes[row, 1].imshow(gt, cmap='gray')
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis('off')

        # Prediction
        axes[row, 2].imshow(pred, cmap='gray')
        axes[row, 2].set_title(f"Prediction\nDice={dice:.3f}")
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.show()




