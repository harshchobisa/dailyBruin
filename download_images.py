import urllib.parse
import urllib.request
import os

import pandas as pd

from PIL import Image
import pandas as pd
import torch
from torchvision import transforms


HEADERS = [
    (
        "User-Agent",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    )
]


def main():
    df = pd.read_csv("./data_new/final_merged.csv")

    for i, row in df.iterrows():
        if pd.notna(row.image_url):
            print(f"Downloading image: {i}/{len(df)}", end="\r")

            page, image_url = row.page, row.image_url

            ext = image_url.split(".")[-1]

            opener = urllib.request.build_opener()
            opener.addheaders = HEADERS
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(
                image_url, f"./images/{urllib.parse.quote(page, safe='')}.{ext}"
            )

            get_image_vector(page, image_url)



def get_image_vector(page, image_url):
    image_ext = image_url.split(".")[-1]
    image_path = f"./images/{urllib.parse.quote(page, safe='')}.{image_ext}"
    processed_image_path = f"./processed_images/{urllib.parse.quote(page, safe='')}.pt"

    if os.path.isfile(processed_image_path):
        input_tensor = torch.load(processed_image_path)
        return input_tensor

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path)
    image = image.convert("RGB")

    input_tensor = preprocess(image)

    torch.save(input_tensor, processed_image_path)



if __name__ == "__main__":
    main()
