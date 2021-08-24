from datetime import datetime
import time
from statistics import mean
import os

import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchvision

from .datasets import DailyBruinDataset
from .networks import Net


def train(
    train_df,
    val_df,
    epochs,
    batch_size,
    summary_path,
    n_summary,
    n_eval,
    layer_sizes,
    dropout_prob,
    learning_rate,
    weight_decay,
    device,
):
    # Create dataloaders
    train_dataset = DailyBruinDataset(train_df)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataset = DailyBruinDataset(val_df)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Create model
    model = Net(layer_sizes=layer_sizes, dropout_prob=dropout_prob, device=device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.CrossEntropyLoss()

    # create summary writers
    summary_writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    models_folder = f"./models/{datetime.now()}"
    os.makedirs("./models", exist_ok=True)
    os.mkdir(models_folder)
    print("Models:", models_folder)

    step = 0
    best_acc = None
    for n in range(epochs):
        print(f"Epoch: {n} of {epochs}")
        start_time = time.time()

        # Keeping track of how long everything takes
        load_start_time = time.time()
        forward_time = 0.0
        backward_time = 0.0

        for i, batch_data in enumerate(train_dataloader):
            load_time = time.time() - load_start_time
            print(
                (
                    f"Iteration: {i} of {len(train_dataloader)}, Load time: {load_time:.3f}, "
                    f"Forward time: {forward_time:.3f}, Backward time: {backward_time:.3f} ..."
                ),
                end="\r",
            )
            x, labels = batch_data
            labels = labels.to(device)

            forward_start_time = time.time()
            outputs = model.forward(x)
            forward_time = time.time() - forward_start_time

            backward_start_time = time.time()
            loss = loss_function(outputs, labels)
            accuracy = compute_accuracy(outputs, labels)
            backward_time = time.time() - backward_start_time

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % n_summary == 0:
                summary_writer.add_scalar("Training Loss", loss, global_step=step)
                summary_writer.add_scalar(
                    "Training Accuracy", accuracy, global_step=step
                )

            if step % n_eval == 0:
                model.eval()
                acc = evaluate(model, val_dataloader, summary_writer, step, device)
                # Save model if it's the best so far
                if best_acc is None or acc > best_acc:
                    torch.save(model.state_dict(), f"{models_folder}/cur.pt")
                    best_acc = acc
                model.train()

            step += 1
            load_start_time = time.time()

        print(f"\nEpoch took {time.time() - start_time} seconds")

    summary_writer.close()
    torch.save(model.state_dict(), f"{models_folder}/final_model.pt")


def compute_accuracy(outputs, labels):
    outputs = outputs.argmax(axis=1)
    return (outputs == labels).sum().item() / len(outputs)


def evaluate(model, val_dataloader, summary_writer, step, device):
    loss_function = nn.CrossEntropyLoss()
    with torch.no_grad():
        # WHOLE SET
        accuracies = []
        losses = []
        for i, batch_data in enumerate(val_dataloader):
            print(f"(val) Iteration {i} of {len(val_dataloader)} ...", end="\r")
            x, labels = batch_data
            labels = labels.to(device)

            outputs = model.forward(x)
            losses.append(loss_function(outputs, labels))
            accuracies.append(compute_accuracy(outputs, labels))

        # Log loss and accuracy
        summary_writer.add_scalar("Validation Loss", sum(losses), global_step=step)
        summary_writer.add_scalar(
            "Validation Accuracy", mean(accuracies), global_step=step
        )

        # SINGLE EXAMPLE
        dataloader_iter = iter(val_dataloader)
        x, labels = dataloader_iter.next()
        labels = labels.to(device)

        outputs = model.forward(x)

        # Log image and texts
        # Shape of image: (4, 3, 224, 224)
        first_images = x["image1"][:4]
        second_images = x["image2"][:4]
        first_texts = x["text1"][:4]
        second_texts = x["text2"][:4]

        # Denormalize images
        first_images[:, 0, :, :] = first_images[:, 0, :, :] * 0.229 + 0.485
        first_images[:, 1, :, :] = first_images[:, 1, :, :] * 0.224 + 0.456
        first_images[:, 2, :, :] = first_images[:, 2, :, :] * 0.225 + 0.406
        second_images[:, 0, :, :] = second_images[:, 0, :, :] * 0.229 + 0.485
        second_images[:, 1, :, :] = second_images[:, 1, :, :] * 0.224 + 0.456
        second_images[:, 2, :, :] = second_images[:, 2, :, :] * 0.225 + 0.406

        summary_writer.add_image(
            "Validation article images",
            torchvision.utils.make_grid(
                torch.cat([first_images, second_images], dim=0), nrow=2
            ),
            global_step=step,
        )
        summary_writer.add_text(
            "Validation article first text", "\n\n".join(first_texts), global_step=step,
        )
        summary_writer.add_text(
            "Validation article second text",
            "\n\n".join(second_texts),
            global_step=step,
        )
        summary_writer.add_text(
            "Validation prediction", str(outputs[:4]), global_step=step
        )
        summary_writer.add_text(
            "Validation expected", str(labels[:4]), global_step=step
        )

    return mean(accuracies)
