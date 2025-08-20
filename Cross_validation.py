import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from SimpleRamanCNN import SimpleRamanCNN
from Raman_Dataset import RamanDataset


# === K-fold cross validation setup ===
def run_kfold_cv(spectra, targets, k=5, batch_size=32, epochs=10,model_class = SimpleRamanCNN):

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    train_losses=[]#arrays containing loss per epoch per model
    val_losses=[]
    models=[]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(spectra)):
        print(f"\n---- Fold {fold+1}/{k} ----")


        train_dataset = RamanDataset(spectra[train_idx], targets[train_idx], train=True)
        val_dataset   = RamanDataset(spectra[val_idx], targets[val_idx], train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        model = model_class()  # replace with your model class
        model = model.to(device) #move to gpu if available
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss(reduction="mean")

        

        for epoch in range(epochs):
            model.train()
            epoch_train_loss=0.0

            for X, y in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            epoch_train_loss  = epoch_train_loss/ len(train_loader)

            train_losses.append(epoch_train_loss) 

            

            # Validation loop
            model.eval()
            epoch_val_loss=0.0
            with torch.no_grad():
                for X, y in val_loader: 
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    out = model(X)
                    epoch_val_loss += criterion(out, y).item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}")
            val_losses.append(avg_val_loss)
        
        #store models
        torch.save(model.state_dict(), f"model{fold+1}.pth")

    return train_losses, val_losses, models


