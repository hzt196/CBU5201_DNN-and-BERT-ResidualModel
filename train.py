import torch
import torch.optim as optim
from model import ResidualClassifier


def train(model, dataloader, epochs=5, lr=0.005, save_path='trained_model.pth'):
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 将模型移动到 GPU（如果可用）

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        model.train()  # 切换到训练模式
        for texts, labels in dataloader:
            # 将数据移动到 GPU（如果可用）
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    # 训练结束后保存模型
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")
    return model
