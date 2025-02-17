import copy
import random

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
from modular_addition import ModularOperationsDataset
DATA_SEED = 599
MODEL_SEED = 999
device='cuda'

if __name__ == "__main__":
    # use random as a placeholder to allow the model to still learn the operator
    operations = (lambda x, y: x + y, lambda x, y: torch.randint(0, 113, x.shape))
    # # use duplicate as a placeholder to allow the model to still learn the operator
    # operations = (lambda x, y: x + y, lambda x, y: x+y)
    # # use no placeholder
    # operations = (lambda x, y: x + y)

    dataset = ModularOperationsDataset(
        base=113,
        train_fraction=0.38,
        operations=operations,
    )

    transformer_config = HookedTransformerConfig(
        n_layers=1,
        n_heads=4,
        d_model=128,
        d_head=32,
        d_mlp=512,
        act_fn="relu",
        normalization_type=None,
        d_vocab=dataset.tokens.max().item() + 1,
        d_vocab_out=dataset.base,
        n_ctx=4,
        init_weights=True,
        seed=MODEL_SEED,
    )

    lr = 1e-3
    wd = 1.0
    betas = (0.9, 0.98)
    num_epochs = 25000
    checkpoints_every = 100

    model = HookedTransformer(transformer_config).to(device)

    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=betas
    )


    def loss_fn(logits, labels):
        if len(logits.shape) == 3:
            logits = logits[:, -1]
        logits = logits.to(torch.float64)
        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
        return -correct_log_probs.mean()


    operations_losses = [{'train_losses': [], 'test_losses': []} for i in range(len(operations))]
    train_losses = []
    test_losses = []
    model_checkpoints = []
    checkpoint_epochs = []
    for epoch in tqdm(range(num_epochs)):
        train_logits = model(dataset.train_data)
        train_loss = loss_fn(train_logits, dataset.train_labels)
        train_loss.backward()
        train_losses.append(train_loss.item())
        optimizer.step()
        optimizer.zero_grad()
        # Update dataset halfway though
        if epoch == round(num_epochs / 2):
            operations = (lambda x, y: x + y, lambda x, y: x - y)
            dataset = ModularOperationsDataset(
                base=113,
                train_fraction=0.38,
                operations=operations,
            )

        with torch.inference_mode():
            test_logits = model(dataset.test_data)
            test_loss = loss_fn(test_logits, dataset.test_labels)
            test_losses.append(test_loss.item())
            for i in range(len(operations)):
                train_logits = model(dataset.train_data[i::len(operations)])
                train_loss = loss_fn(train_logits, dataset.train_labels[i::len(operations)])
                operations_losses[i]['train_losses'].append(train_loss.item())
                test_logits = model(dataset.test_data[i::len(operations)])
                test_loss = loss_fn(test_logits, dataset.test_labels[i::len(operations)])
                operations_losses[i]['test_losses'].append(test_loss.item())

        if ((epoch + 1) % checkpoints_every) == 0:
            checkpoint_epochs.append(epoch)
            model_checkpoints.append(copy.deepcopy(model.state_dict()))
            print(
                f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}"
            )

    torch.save(
        {
            "model": model.state_dict(),
            "config": model.cfg,
            "checkpoints": model_checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
            "test_losses": test_losses,
            "train_losses": train_losses,
            "train_data": dataset.train_data,
            "test_data": dataset.test_data,
            "operations_losses": operations_losses
        },
        "output_plus_and_minus_staged.pt",
    )

