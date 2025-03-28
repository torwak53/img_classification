# Toto je script pro trenink modelu na clasifikaci obrazků. Data byla stažena z Kaggle.com.
# březen 2025 - Michal Vajskebr

# Import knihoven
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
from timeit import default_timer as timer
import requests
import os
from os import rmdir

# nastavení defaulního zařízení
device = "cuda" if torch.cuda.is_available() else "cpu"

# nastavení složek kde se nachází data
DATA_PATH = Path("data/ovocko")
train_dir = DATA_PATH / "train"  # pro data na trenink
test_dir = DATA_PATH / "test"  # pro data na test

# nastavení transformace pro obrázky na trenink
train_trans = transforms.Compose([transforms.Resize(size=(32, 32)),  # snížení rozlišení obrázku na 32*32 pixelů
                                  transforms.TrivialAugmentWide(num_magnitude_bins=20),  # náhodné upravení obrázku
                                  transforms.ToTensor()])  # převedení na tenzor

# nastavení transformace pro obrázky na test
test_trans = transforms.Compose([transforms.Resize(size=(32, 32)),  # snížení rozlišení obrázku na 32*32 pixelů
                                 transforms.ToTensor()])  # převedení na tenzor

# upravení našich obrázků na čísla podle předchozího nastavení
train_data = datasets.ImageFolder(root=train_dir, transform=train_trans)  # pro data na trenink
test_data = datasets.ImageFolder(root=test_dir, transform=test_trans)  # pro data na test

# rozdělení dat do dávek
BATCH_SIZE = 32  # velikost jedné dávky
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  # zamíchání a rozdělení trenink dat
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)  # rozdělení test dat


# model na trénování
class FruitImgClassification(nn.Module):  # Model je podmodelem nn.Module
    def __init__(self, input_num, hidden_units, output_num):
        super().__init__()
        self.block1 = nn.Sequential(  # první blok s více vrstvami pro trénink
            nn.Conv2d(in_channels=input_num, out_channels=hidden_units, kernel_size=3),  # zkoumá data obrázku po 3*3 pixelech
            nn.ReLU(),  # ze všech záporných hodnot udělá 0
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # zkoumá nejvyššší hodnotu na 2*2 pixelech
        )
        self.block2 = nn.Sequential(  # druhý blok s více vrstvami pro trénink
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier_layer = nn.Sequential(
            nn.Flatten(),  # převedení tenzoru na list hodnot
            nn.Linear(in_features=hidden_units*5**2, out_features=output_num),
            nn.Dropout(p=0.5)  # smazání 50% dat
        )

    def forward(self, x):  # provedení dat postupně všemi třemi bloky
        return self.classifier_layer(self.block2(self.block1(x)))

classes = train_data.classes  # nastavení možných výsledků obrázku
print(classes)

# uložení našeho modelu do proměnné s určitými hodnotami a na defaulním zařízení
model = FruitImgClassification(input_num=3, hidden_units=16, output_num=len(classes)).to(device)

loss_fn = nn.CrossEntropyLoss()  # nastavení funkce pro výpočet ztráty
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)  # nastavení optimalizátoru a jeho parametrů

# '''
# funkce pro trénink
def train_step(data_loader, model, loss_fn, optimizer, device=device):
    model.train()  # určujeme že model bude trénovat
    train_loss, train_acc = 0, 0  # příprava proměnných pro následný výopočet ztráty a přesnosti
    for X, y in data_loader:  # opakování pro všechny prvky v dávce kde X je obrázek a y je požadovaný výsledek
        X, y = X.to(device), y.to(device)  # přesun proměnných na def. zařízení
        logits = model(X)  # spuštění modelu s daty s obrázku a výstup uložen do proměnné
        labels = torch.argmax(logits, dim=1)  # z výstupu vybrána nejvyšší hodnota a uložena jako výsledek
        loss = loss_fn(logits, y)  # výpočet aktuální ztáty mezi výstupem a správným řešením
        train_loss += loss.item()  # připočtení aktuální ztráty k celkové
        acc = (labels == y).sum().item()/len(labels)  # výpčet aktuální přesnosti pomocí vzorečku pro přesnost z výsledku a správného řešení
        train_acc += acc  # připočtení aktuální přesnosti k celkové
        # kroky pro trénink modelu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)  # výpočet průměrné ztráty z jedné dávky
    train_acc /= len(data_loader)  # výpočet průměrné přesnosti z jedné dávky
    return train_loss, train_acc  # vrácení zprůměrovaných hodnot


# funkce pro testování
def test_step(data_loader, model, loss_fn, device=device):
    model.eval()  # model budeme testovat
    # postup stejný jako u fce pro trénovaní až na samotne trenovaní
    test_loss, test_acc = 0, 0
    for X_test, y_test in data_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        test_logits = model(X_test)
        test_labels = torch.argmax(test_logits, dim=1)
        loss = loss_fn(test_logits, y_test)
        test_loss += loss.item()
        acc = (test_labels == y_test).sum().item()/len(test_labels)
        test_acc += acc
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    return test_loss, test_acc


# spuštění obou funkcí s výše nastavenými daty
def train_and_test(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs):
    train_loss, train_acc, test_loss, test_acc = [], [], [], []  # připravení polí pro ztátu a přesnostu u treninku i u testu
    for epoch in range(epochs):  # spuštění obou funkcí s konkrétním opakováním
        train_loss_value, train_acc_value = train_step(data_loader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)  # fce pro trenovaní a vrácené hodnoty uloženy v proměnných
        test_loss_value, test_acc_value = test_step(data_loader=test_dataloader, model=model, loss_fn=loss_fn)  # fce pro testování a vrácené hodnoty uloženy v proměnných
        # uložení vrácených hodnot do polí
        train_loss.append(train_loss_value)
        train_acc.append(train_acc_value)
        test_loss.append(test_loss_value)
        test_acc.append(test_acc_value)
        print(f"Epoch: {epoch} | Train acc: {100*train_acc_value:.2f}% | Test acc: {100*test_acc_value:.2f}%")  # zobrazení vrácené přesnosti
    return train_loss, train_acc, test_loss, test_acc  # vrácení polí pro zobrazení


EPOCHS = 20  # nastavení počtu opakování
start_time = timer()  # spuštění časomíry
# uložení vrácených hodnot (polí) z funkce pro opakování do proměnných
train_loss, train_acc, test_loss, test_acc = train_and_test(train_dataloader=train_dataloader, test_dataloader=test_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=EPOCHS)
end_time = timer()  # konec časomíry
print(f"Final time: {end_time-start_time}s")  # zobrazení času trénování a testování v sekundách


# fce pro zobrazení grafu ztráty a přesnosti v průběhu opakování
def plot_data(train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc):
    epochs = range(len(train_loss))  # nastavení rozsahu opakování
    plt.figure(figsize=(10, 7))  # nastavení velikosti obrázku

    plt.subplot(1, 2, 1)  # budeme nastavovat grafy v 1 řádku, 2 sloupcích, momentálně první (1)
    plt.plot(epochs, train_loss, label="train loss")  # zobrazení ztáty u tréninku
    plt.plot(epochs, test_loss, label="test loss")  # zobrazení ztáty u testování
    plt.title("LOSS")  # natavení názvu grafu
    plt.xlabel("epochs")  # nastavení štítku u osy x
    plt.legend()  # zobrazení legendy

    plt.subplot(1, 2, 2)  # stejné jako u prvního grafu, tentokrát u grafu číslo 2
    plt.plot(epochs, train_acc, label="train acc")  # přesnost tréninku
    plt.plot(epochs, test_acc, label="test acc")  # přesnost testů
    plt.title("Accuracy")  # natavení názvu grafu
    plt.xlabel("epochs")  # nastavení štítku u osy x
    plt.legend()  # zobrazení legendy

    plt.show()  # zobrazení celého obrázku


plot_data()  # volání fce pro zobrazení
# '''
models_dir = Path("models")
model_name = "FruityModel.pth"
model_save_path = models_dir / model_name

torch.save(obj=model.state_dict(), f=model_save_path)
print(f"Model saved to {model_save_path}.")

loaded_model_path = model_save_path
loaded_model = FruitImgClassification(input_num=3, hidden_units=16, output_num=len(classes))
loaded_model.load_state_dict(torch.load(f=loaded_model_path))


# def own_img():  # při žádosti získat obrázek podle url adresy
#     link = input("Zadejte odkaz na obrázek: ")
#     name = "auto"
#     path = Path("pic") / name
#     with open(path, "wb") as f:
#         req = requests.get(link)
#         f.write(req.content)
#     return path


# divný soubor od google colab, vytvořený po smazání souboru
if os.path.exists("pic/.ipynb_checkpoints"):
    rmdir(Path("pic/.ipynb_checkpoints"))  # smazání aby nedělal nepořádek


def get_image(directory):  # funkce pro získání místa uloženého obrázku
    image_files = list(directory.glob("*"))  # načte obrázky do listu
    return image_files[0] if image_files else None  # vrátí první (jediný) z nich


image_path = get_image(Path("pic"))  # vrácenou cestu k obrázku uloží do proměnné

# image_path = own_img()  # pro jiný způsob
img_unit8 = torchvision.io.read_image(path=image_path)  # přečte obrázek a uloží do proměnné
img_unit8 = img_unit8[:3, :, :]  # při 4 barevných kanálech se odstraní průhlednost
img = img_unit8.type(torch.float32) / 255  # data obrázku se vydělí pro získání dat v intervalu <0;1>


transform = transforms.Resize(size=(32, 32))  # nastavení transformace pro obrázek - zmenšení na 32*32 pixelů
img_resized = transform(img)  # transformace se aplikuje na obrázek

loaded_model.eval()  # budeme zhodnocovat model
with torch.inference_mode():
    preds = loaded_model(img_resized.unsqueeze(dim=0))  # proběhneme náš obrázek skrz model a výsledky uložíme do proměnné
    probs = torch.softmax(preds, dim=1)
    label = torch.argmax(probs, dim=1)  # zjistíme největší hodnotu z nich, tedy nejpravděpodobnější odpověď
    print(classes[label])  # Odpověď nakonec zobrazíme


plt.imshow(img_resized.permute(1, 2, 0)),  # můžeme zobrazit i zmenšený obrázek
plt.axis(False)  # vypneme osy x a y
plt.show()  # zobrazíme

Path.unlink(image_path)  # pro příští použití obrázek smažeme
