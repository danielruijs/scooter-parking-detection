from pydantic import BaseModel


class VisionModel(BaseModel):
    name: str
    hf_id: str
    display_name: str


MODELS: dict[str, VisionModel] = {
    "vit": VisionModel(
        name="vit", hf_id="google/vit-base-patch16-224", display_name="ViT (Base)"
    ),
    "resnet18": VisionModel(
        name="resnet18", hf_id="microsoft/resnet-18", display_name="ResNet-18"
    ),
    "mobilenetv3": VisionModel(
        name="mobilenetv3",
        hf_id="timm/mobilenetv3_small_100.lamb_in1k",
        display_name="MobileNetV3 (Small)",
    ),
    "dinov2": VisionModel(
        name="dinov2", hf_id="facebook/dinov2-small", display_name="DINOv2 (Small)"
    ),
    "efficientnetb0": VisionModel(
        name="efficientnetb0",
        hf_id="google/efficientnet-b0",
        display_name="EfficientNet-B0",
    ),
    "convnextv2": VisionModel(
        name="convnextv2",
        hf_id="facebook/convnextv2-tiny-1k-224",
        display_name="ConvNeXt V2 (Tiny)",
    ),
}
