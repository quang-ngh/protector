from PIL import Image
import torchvision.transforms as transforms

def read_image(src, pixel_range = (0, 1), add_batch_dimension = True):
    """

    """
    img = Image.open(src).convert('RGB')

    modules = [transforms.ToTensor()]

    if pixel_range == (-1, 1):
        modules.append(transforms.Normalize([0.5], [0.5]))
    
    pipe = transforms.Compose(modules)
    
    torch_img = pipe(image)

    if add_batch_dimension:
        torch_img = torch_img.unsqueeze(0)

    return torch_img