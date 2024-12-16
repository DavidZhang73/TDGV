"""Modified based on https://github.com/microsoft/VideoX/blob/master/2D-TAN/lib/datasets/__init__.py#L30-L43."""

import torch
from torch import Tensor, nn


def average_to_fixed_length(visual_input, num_sample_clips):
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips
    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input


def sample_to_fixed_length_list(
    features: list[Tensor],  # list of B, each is N_i x D
    length: int,  # -1 return the original features
):
    if length == -1:
        return features
    new_features_list = []
    for feature in features:
        new_features_list.append(average_to_fixed_length(feature, length))
    return torch.stack(new_features_list, dim=0)


def sample_to_fixed_length(
    features: Tensor,  # B x N x D
    input_mask: Tensor,  # Contains the length of each sample in the batch
    length: int,  # -1 return the original features
):
    if length == -1:
        return features
    B, _, _ = features.shape
    new_features_list = []
    for i in range(B):
        feature_length = (~input_mask[i]).sum().item()
        new_features_list.append(average_to_fixed_length(features[i, :feature_length], length))
    return nn.utils.rnn.pad_sequence(new_features_list, batch_first=True)
