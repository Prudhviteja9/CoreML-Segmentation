import torch
import torchvision
import coremltools as ct

class SegmentationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: (1, 3, H, W) normalized
        out = self.model(x)["out"]  # (1, C, H, W)
        return out

def get_model(device):
    base_model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    base_model.eval()
    base_model.to(device)
    return SegmentationWrapper(base_model).to(device)

def main():
    device = "cpu"  # CoreML export works best on CPU
    model = get_model(device)

    # Example input with fixed size
    example_input = torch.randn(1, 3, 224, 224, device=device)

    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input_image",
                             shape=example_input.shape,
                             scale=1/255.0,
                             bias=[0, 0, 0])]
    )

    mlmodel.save("models/segmentation_deeplabv3.mlmodel")
    print("CoreML model saved to models/segmentation_deeplabv3.mlmodel")

if __name__ == "__main__":
    main()
