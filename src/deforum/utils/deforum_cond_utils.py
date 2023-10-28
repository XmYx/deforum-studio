from clip_interrogator import Config, Interrogator
import torch.nn.functional as F
import torch



def calculate_global_average(conds):
    """
    Calculate the global average for both tensor and pooled_tensor across all conditions.

    Args:
    - conds (list): List of conditionings.

    Returns:
    - Tuple containing global averages for tensor and pooled_tensor respectively.
    """
    total_tensors = len(conds)
    sum_tensor = torch.zeros_like(conds[0][0][0])
    sum_pooled_tensor = torch.zeros_like(conds[0][0][1]['pooled_output'])

    for cond in conds:
        sum_tensor += cond[0][0]
        sum_pooled_tensor += cond[0][1]['pooled_output']

    avg_tensor = sum_tensor / total_tensors
    avg_pooled_tensor = sum_pooled_tensor / total_tensors

    return avg_tensor, avg_pooled_tensor

def pad_tensor_to_match_size(tensor1, tensor2):
    """
    Pad tensor1 or tensor2 (whichever is smaller in the second dimension) to match the size of the other.
    Fills the newly created empty area with the other tensor's data.
    """
    d1, d2 = tensor1.size(1), tensor2.size(1)
    diff = d2 - d1

    if diff > 0:  # tensor1 is smaller, pad it
        # Get a slice from tensor2 and append to tensor1
        slice_from_tensor2 = tensor2[:, :diff]
        tensor1 = torch.cat((tensor1, slice_from_tensor2), dim=1)
    elif diff < 0:  # tensor2 is smaller, pad it
        # Get a slice from tensor1 and append to tensor2
        slice_from_tensor1 = tensor1[:, :abs(diff)]
        tensor2 = torch.cat((tensor2, slice_from_tensor1), dim=1)

    return tensor1, tensor2
def pyramid_blend(tensor1, tensor2, blend_value):
    # For simplicity, we'll use two levels of blending
    downsampled1 = F.avg_pool2d(tensor1, 2)
    downsampled2 = F.avg_pool2d(tensor2, 2)

    blended_low = (1 - blend_value) * downsampled1 + blend_value * downsampled2
    blended_high = tensor1 + tensor2 - F.interpolate(blended_low, scale_factor=2)

    return blended_high


def gaussian_blend(tensor1, tensor2, blend_value):
    sigma = 0.5  # Adjust for desired smoothness
    weight = torch.exp(-((blend_value - 0.5) ** 2) / (2 * sigma ** 2))

    return (1 - weight) * tensor1 + weight * tensor2
def sigmoidal_blend(tensor1, tensor2, blend_value):
    # Convert blend_value into a tensor with the same shape as tensor1 and tensor2
    blend_tensor = torch.full_like(tensor1, blend_value)
    weight = 1 / (1 + torch.exp(-10 * (blend_tensor - 0.5)))  # Sigmoid function centered at 0.5
    return (1 - weight) * tensor1 + weight * tensor2

def blend_tensors(obj1, obj2, blend_value, blend_method="linear"):
    """
    Blends tensors in two given objects based on a blend value using various blending strategies.
    """

    tensor1, tensor2 = pad_tensor_to_match_size(obj1[0], obj2[0])
    pooled_tensor1, pooled_tensor2 = pad_tensor_to_match_size(obj1[1]['pooled_output'], obj2[1]['pooled_output'])

    if blend_method == "linear":
        weight = blend_value
        blended_cond = (1 - weight) * tensor1 + weight * tensor2
        blended_pooled = (1 - weight) * pooled_tensor1 + weight * pooled_tensor2

    elif blend_method == "sigmoidal":
        blended_cond = sigmoidal_blend(tensor1, tensor2, blend_value)
        blended_pooled = sigmoidal_blend(pooled_tensor1, pooled_tensor2, blend_value)

    elif blend_method == "gaussian":
        blended_cond = gaussian_blend(tensor1, tensor2, blend_value)
        blended_pooled = gaussian_blend(pooled_tensor1, pooled_tensor2, blend_value)

    elif blend_method == "pyramid":
        blended_cond = pyramid_blend(tensor1, tensor2, blend_value)
        blended_pooled = pyramid_blend(pooled_tensor1, pooled_tensor2, blend_value)

    return [[blended_cond, {"pooled_output": blended_pooled}]]

# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res
