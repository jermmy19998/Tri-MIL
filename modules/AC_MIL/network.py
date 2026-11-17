import os
import torch
import torch.nn as nn


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        """Initialize the Classifier_1fc instance and configure its dependencies."""
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):
        """Compute the Classifier_1fc forward pass for the supplied input tensors."""
        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        """Initialize the residual_block instance and configure its dependencies."""
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Compute the residual_block forward pass for the supplied input tensors."""
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        """Initialize the DimReduction instance and configure its dependencies."""
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):
        """Compute the DimReduction forward pass for the supplied input tensors."""
        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class DimReduction1(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        """Initialize the DimReduction1 instance and configure its dependencies."""
        super(DimReduction1, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):
        """Compute the DimReduction1 forward pass for the supplied input tensors."""
        x_ = x
        x = self.fc1(x)
        x = self.relu1(x + x_)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x
