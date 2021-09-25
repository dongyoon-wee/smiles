import torch


def train_step(batch_item, training, model, optimizer, criterion, device=torch.device("cpu")):
    # img = batch_item['img'].to(device)
    seq = batch_item['seq'].to(device)
    label = batch_item['label'].to(device)
    if training is True:
        model.train()
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        output = model(seq)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        return loss
    else:
        model.eval()
        with torch.no_grad():
            output = model(seq)
            loss = criterion(output, label)

        return loss