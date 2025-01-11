import torch

def train_model(model, dataloader, criterion, optimizer, num_epochs, vocab_size, device):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Epoch {epoch+1} of {num_epochs} running")
        for batch_source, batch_target in dataloader:
            batch_source, batch_target = batch_source.to(device), batch_target.to(device)

            optimizer.zero_grad()
            output = model(batch_source, batch_target[:, :-1])
            output = output.reshape(-1, vocab_size)
            target = batch_target[:, 1:].reshape(-1)
            loss = criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #gradient clipping
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
