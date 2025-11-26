import warnings

import timm
from pytorch_transformers import BertConfig, BertForSequenceClassification

from method.models.resnet import resnet18


def get_model(model_name: str, num_classes: int, size: int, pretrained: bool = True):
    if model_name == "resnet18" and size == 32:
        return resnet18(num_classes=num_classes)
    elif "vit" in model_name or "resnet" in model_name:
        return timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)
    elif "bert" in model_name:
        config = BertConfig.from_pretrained(
            "bert-base-uncased", num_labels=3, finetuning_task="mnli"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", from_tf=False, config=config
            )
        return model
