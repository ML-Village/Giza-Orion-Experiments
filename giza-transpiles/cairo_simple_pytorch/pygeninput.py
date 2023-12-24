import torch
import torchvision
#import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

def to_fixed_point(val, bits):
    return round(val * (2**bits))

def mnist_image_to_fixed_point(data):
    #print(type(data[0]))
    return [to_fixed_point(val.item(), 16) for val in data]


def generate_input_cairo(data):
    values = mnist_image_to_fixed_point(data)
    print(values, "\n", "\n")
    values = [f"FixedTrait::<FP16x16>::new({val}, {'true' if val < 0 else 'false'})" for val in values]
    return ",\n ".join(values)

# download = len(os.listdir('./../../data')) == 0 # check if need to download data
# print(f"Downloading data: {download}")
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./../../data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Resize((14,14)),
                                    torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                                ])), shuffle=True)

(example_data, example_targets) = next(iter(test_loader)) #1 item of 196 flattened pixels

print(f"example_data.shape: {example_data.shape}")
print(f"example_targets.shape: {example_targets.shape}")
#print(example_data[0]) # shape is 1,196
#print(example_data[1]) # there is no 1 only 0

input_cairo = generate_input_cairo(example_data[0]) # list of 196 fixed point values
print(input_cairo)

with open("./src/input.cairo", "w") as f:
    f.write("""
        use array::{SpanTrait, ArrayTrait};
        use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
        use orion::numbers::{FixedTrait, FP16x16};
        fn input() -> Tensor<FP16x16> {
            TensorTrait::<FP16x16>::new(
                array![196].span(),
                array![
            """)
    f.write(input_cairo)
    f.write("""
                ].span()
            )
        }
            """)