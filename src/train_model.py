import os, torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from model import BreastCancerCNN
from data_loader import train_loader, val_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = BreastCancerCNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("models", exist_ok=True)
    best = 0.0
    for e in range(10):
        model.train(); tl, tc, tot = 0.0, 0, 0
        for x,y in tqdm(train_loader, desc=f"Epoch {e+1}/10 [Train]", leave=False):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x); loss = crit(out,y)
            loss.backward(); opt.step()
            tl += loss.item()
            tc += (out.argmax(1)==y).sum().item(); tot += y.size(0)
        ta = tc/tot

        model.eval(); vl, vc, vot = 0.0, 0, 0
        with torch.no_grad():
            for x,y in tqdm(val_loader, desc=f"Epoch {e+1}/10 [Val]", leave=False):
                x,y = x.to(device), y.to(device)
                out = model(x); loss = crit(out,y)
                vl += loss.item()
                vc += (out.argmax(1)==y).sum().item(); vot += y.size(0)
        va = vc/vot
        print(f"\nEpoch {e+1}/10")
        print(f"Train Loss {tl/len(train_loader):.4f} | Train Acc {ta*100:.2f}%")
        print(f"Val   Loss {vl/len(val_loader):.4f} | Val   Acc {va*100:.2f}%")

        if va>best:
            best=va; torch.save(model.state_dict(),"models/best_model.pth")
            print("✅ Best model saved!")
    print("Done. Best Val Acc:", best*100)

if __name__=="__main__":
    main()
