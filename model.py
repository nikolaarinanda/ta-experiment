import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNLight(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=2):
        super(TextCNNLight, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, 64, kernel_size=4)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = F.relu(self.conv1(x)).max(dim=2)[0]
        x2 = F.relu(self.conv2(x)).max(dim=2)[0]
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TextCNNMedium(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, 100, kernel_size=4)
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=5)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = F.relu(self.conv1(x)).max(dim=2)[0]
        x2 = F.relu(self.conv2(x)).max(dim=2)[0]
        x3 = F.relu(self.conv3(x)).max(dim=2)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TextCNNHeavy(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, num_classes=2):
        super(TextCNNHeavy, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=4),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=5),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = self.conv1(x).max(dim=2)[0]
        x2 = self.conv2(x).max(dim=2)[0]
        x3 = self.conv3(x).max(dim=2)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
