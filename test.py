import torch
from sklearn.metrics import accuracy_score
from model import ResidualClassifier

def test(dataloader, input_size, model_path='trained_model.pth'):
    # 初始化模型并加载权重
    model = ResidualClassifier(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换到评估模式

    all_preds, all_labels = [], []

    with torch.no_grad():
        for texts, labels in dataloader:
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy
