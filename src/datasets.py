import os
import urllib.parse

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms


class DailyBruinDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

        # Create processed_images directory
        os.makedirs("./processed_images", exist_ok=True)

        self.indices = []
        for i in range(len(data)):
            for j in range(len(data)):
                if i == j:
                    continue
                self.indices.append((i, j))

        self.list_data = []
        for i, row in self.data.iterrows():
            print(f"Row {i}", end="\r")
            row_data = get_row_data(row)
            self.list_data.append(row_data)

    def __getitem__(self, index):
        i, j = self.indices[index]

        image1, text1, length1, categories1, days_posted1, views1 = self.list_data[i]
        image2, text2, length2, categories2, days_posted2, views2 = self.list_data[j]

        # 0 if first article has more views
        output = int(views1 < views2)

        input_data = {
            "image1": image1,
            "text1": text1,
            "length1": length1,
            "categories1": categories1,
            "days_posted1": days_posted1,
            "views1": views1,
            "image2": image2,
            "text2": text2,
            "length2": length2,
            "categories2": categories2,
            "days_posted2": days_posted2,
            "views2": views2,
        }

        return (input_data, output)

    def __len__(self):
        return len(self.indices)


def get_row_data(row):
    page, image_url, title, length, categories, days_posted, views = (
        row.page,
        row.image_url,
        row.title,
        row.length,
        row.categories_one_hot,
        row.n_days_posted,
        row.views,
    )
    image = get_image_vector(page, image_url)
    categories = torch.tensor(list(map(int, categories[1:-1].split(", "))))
    # [news, arts-entertainment, sports, opinion, campus, opinion-columns, ae-columns, los-angeles, quad, music]
    top_10_categories = categories[[49, 7, 62, 52, 16, 53, 4, 39, 57, 47]]

    return image, title, length, top_10_categories, days_posted, views


def get_image_vector(page, image_url):

    processed_image_path = f"./processed_images/{urllib.parse.quote(page, safe='')}.pt"
    input_tensor = torch.load(processed_image_path)

    return input_tensor


def load_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.dropna(subset=["image_url", "encoded_text", "views"])
    data = data.reset_index(drop=True)
    return data
