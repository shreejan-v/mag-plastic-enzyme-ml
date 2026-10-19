import numpy as np, torch, torch.nn as nn
class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(21, 32, padding_idx=0)
        s.c1 = nn.Conv1d(32, 128, 9, padding=4); s.c2 = nn.Conv1d(128, 128, 9, padding=4)
        s.drop = nn.Dropout(0.3); s.fc = nn.Linear(128, 1)
    def forward(s, x):
        h = s.emb(x).transpose(1, 2)
        h = torch.relu(s.c2(torch.relu(s.c1(h))))
        return s.fc(s.drop(h.max(2).values)).squeeze(1)
def fit_cnn(xtr, ytr, xte, spw, epochs=12):
    m = CNN(); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(spw) ** 0.5))
    xt, yt = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            b = perm[i:i + 64]; opt.zero_grad(); lossf(m(xt[b]), yt[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(torch.cat([m(torch.tensor(xte[i:i + 256])) for i in range(0, len(xte), 256)])).numpy()

