import pandas as pd 
import torch

if __name__ == "__main__":
    test_df = pd.read_csv("input/test.csv")

    # normalize the data
    test_data = test_df / 255.0

    # split for test data
    test_images_tensor = torch.from_numpy(test_data.to_numpy()).float()

    # dataloader
    test_loader = torch.utils.data.DataLoader(
        test_images_tensor, batch_size=300, num_workers=2, shuffle=False
    )

    # load model
    model = torch.load("models/model_ann.pt")

    model.eval()

    test_preds = torch.LongTensor()

    for i, data in enumerate(test_loader):
        
        data = data.to("cuda")
        
        output = model(data)

        preds = output.to("cpu").argmax(axis=1)

        test_preds = torch.cat((test_preds, preds), dim=0)
    
    sample = pd.read_csv("input/sample_submission.csv")

    sample["Label"] = test_preds.numpy()

    sample.to_csv("data/final.csv", index=False)
