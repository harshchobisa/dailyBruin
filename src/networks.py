import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, layer_sizes=[256, 128, 2], dropout_prob=None, device=None):
        super(Net, self).__init__()
        self.device = device

        if dropout_prob is not None and dropout_prob > 0.5:
            print("Are you sure dropout_prob is supposed to be greater than 0.5?")

        # Load Roberta
        self.roberta = torch.hub.load(
            "pytorch/fairseq", "roberta.base", pretrained=True
        )
        for param in self.roberta.parameters():
            param.requires_grad = False
        self.roberta.eval()

        # Load ResNet
        resnet_full = torch.hub.load(
            "pytorch/vision:v0.6.0", "resnet18", pretrained=True
        )
        self.resnet = torch.nn.Sequential(*list(resnet_full.children())[:-1])
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        # self.resnet.eval()

        # self.lstm = nn.LSTM(input_size=768, hidden_size=768 * 2)
        # self.lstm.eval()

        # Layers
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.drops = None if dropout_prob is None else nn.ModuleList()
        prev_size = 2 * 512 + 2 * 768 + 2 * 10 + 2 * 2
        for i, size in enumerate(layer_sizes):
            self.bns.append(nn.BatchNorm1d(prev_size))
            self.fcs.append(nn.Linear(prev_size, size))
            if dropout_prob is not None:
                self.drops.append(nn.Dropout(p=dropout_prob))
            prev_size = size

    def forward(self, inputs):
        first_images = inputs["image1"].to(self.device)
        first_text = inputs["text1"]
        first_length = inputs["length1"].to(self.device)
        first_categories = inputs["categories1"].to(self.device)
        first_days_posted = inputs["days_posted1"].to(self.device)

        second_images = inputs["image2"].to(self.device)
        second_text = inputs["text2"]
        second_length = inputs["length2"].to(self.device)
        second_categories = inputs["categories2"].to(self.device)
        second_days_posted = inputs["days_posted2"].to(self.device)

        # Resnet
        image_tensor_one = self.resnet.forward(first_images)
        image_tensor_two = self.resnet.forward(second_images)
        # Roberta
        text_features1 = torch.Tensor()
        text_features2 = torch.Tensor()
        text_features1 = text_features1.to(self.device)
        text_features2 = text_features2.to(self.device)
        for text in first_text:
            first_tokens = self.roberta.encode(text)[:512]
            features = self.roberta.extract_features(first_tokens)
            feature_means = torch.mean(features, dim=1)
            # features = torch.reshape(features, (-1, 1,768))
            # output, (hn, cn) = self.lstm(features)
            # cn = torch.reshape(cn, (1, 768 * 2))
            text_features1 = torch.cat([text_features1, feature_means])
        for text in second_text:
            second_tokens = self.roberta.encode(text)[:512]
            features = self.roberta.extract_features(second_tokens)
            # print("DIMENSION OF FEATURES ", features.shape)
            feature_means = torch.mean(features, dim=1)
            # features = torch.reshape(features, (-1, 1,768))
            # output, (hn, cn) = self.lstm(features)
            # cn = torch.reshape(cn, (1, 768 * 2))
            # print("DIMENSION OF FEATURES ", features.shape)
            text_features2 = torch.cat([text_features2, feature_means])

        # Concatenated tensor
        concat_tensor = torch.cat((image_tensor_one, image_tensor_two), 1)
        concat_tensor = torch.squeeze(concat_tensor)
        concat_tensor = torch.cat((text_features1, text_features2, concat_tensor), 1)
        additional_features = torch.cat(
            [
                torch.reshape(first_length, (-1, 1)),
                torch.reshape(second_length, (-1, 1)),
                torch.reshape(first_days_posted, (-1, 1)),
                torch.reshape(second_days_posted, (-1, 1)),
            ],
            dim=1,
        )
        concat_tensor = torch.cat(
            [
                concat_tensor,
                additional_features.float(),
                first_categories.float(),
                second_categories.float(),
            ],
            dim=1,
        )

        x = concat_tensor
        zipped_layers = (
            zip(self.bns, self.fcs, [None] * len(self.bns))
            if self.drops is None
            else zip(self.bns, self.fcs, self.drops)
        )
        for i, (bn, fc, drop) in enumerate(zipped_layers):
            x = bn(x)
            if drop is not None:
                x = drop(x)
            if i == len(self.bns) - 1:
                x = fc(x)
            else:
                x = F.relu(fc(x))

        return x
