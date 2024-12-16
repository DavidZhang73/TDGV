import numpy as np
import torch
from PIL import Image, ImageDraw

colors = [
    (245, 245, 245),
    (0, 127, 255),
    (224, 139, 6),
    (232, 5, 172),
    (16, 230, 133),
    (2, 3, 205),
    (0, 113, 23),
    (255, 127, 255),
    (252, 205, 25),
    (0, 255, 255),
    (0, 127, 127),
    (183, 8, 229),
    (241, 12, 71),
    (4, 178, 222),
    (1, 187, 20),
    (134, 11, 106),
    (244, 126, 122),
    (231, 198, 184),
    (127, 0, 0),
    (23, 29, 76),
    (48, 240, 10),
    (255, 0, 255),
    (155, 139, 166),
    (255, 255, 0),
    (153, 107, 0),
    (139, 121, 97),
    (11, 88, 188),
    (254, 185, 12),
    (193, 80, 254),
    (127, 106, 187),
    (132, 252, 26),
    (89, 200, 207),
    (122, 252, 220),
    (89, 187, 88),
    (255, 0, 0),
    (154, 41, 51),
    (209, 120, 61),
    (9, 53, 0),
    (249, 133, 190),
    (144, 1, 163),
    (0, 97, 80),
    (204, 10, 117),
    (189, 117, 236),
    (244, 157, 71),
    (249, 176, 129),
    (2, 130, 203),
    (81, 35, 59),
    (7, 172, 132),
    (248, 54, 133),
    (73, 138, 38),
    (250, 57, 208),
    (152, 59, 151),
    (161, 198, 9),
    (37, 67, 117),
    (27, 153, 254),
    (241, 182, 244),
    (0, 164, 79),
    (63, 253, 162),
    (247, 237, 182),
    (107, 94, 96),
    (57, 200, 252),
    (207, 3, 209),
    (144, 177, 140),
    (196, 16, 5),
    (72, 95, 170),
    (34, 165, 180),
    (154, 212, 79),
    (106, 79, 2),
    (233, 224, 73),
    (128, 222, 245),
    (61, 11, 251),
    (123, 172, 3),
    (175, 89, 147),
    (227, 90, 33),
    (144, 200, 179),
    (34, 93, 248),
    (182, 129, 127),
    (34, 11, 32),
    (139, 216, 134),
    (11, 253, 28),
    (46, 198, 166),
    (58, 4, 143),
    (14, 220, 10),
    (217, 124, 216),
    (176, 83, 67),
    (241, 218, 252),
    (86, 138, 214),
    (163, 248, 118),
    (9, 133, 71),
    (202, 69, 230),
    (12, 243, 197),
    (155, 182, 58),
    (27, 77, 25),
    (169, 162, 245),
    (126, 94, 31),
    (72, 84, 121),
    (12, 202, 114),
    (103, 158, 45),
    (79, 88, 58),
    (163, 224, 11),
]


def plot_attention(attention: torch.Tensor):
    attention = attention.detach().cpu()
    return Image.fromarray((attention.clip(0, 1) * 255).numpy().astype(np.uint8))


def plot_similarity_matrix(
    similarity_matrix: torch.Tensor,
    min_max_normalization: bool = True,
):
    similarity_matrix = similarity_matrix.detach().cpu()
    # min max normalization
    if min_max_normalization:
        similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (
            similarity_matrix.max() - similarity_matrix.min()
        )
    # content
    return Image.fromarray((similarity_matrix.clip(0, 1) * 255).numpy().astype(np.uint8))


def plot_temporal(
    labels: torch.Tensor,
    bar_height: int = 50,
):
    labels = labels.detach().cpu()
    length = len(labels)
    canvas_width = length
    canvas_height = bar_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(labels):
        draw.rectangle(
            (i, 0, i + 1, bar_height),
            fill=colors[label],
            outline=None,
        )
    return canvas


def plot_temporal_alignment(
    preds: torch.Tensor,
    ground_truth: torch.Tensor,
    bar_height: int = 50,
    bar_gap: int = 20,
):
    preds_image = plot_temporal(preds, bar_height)
    ground_truth_image = plot_temporal(ground_truth, bar_height)
    canvas_width = preds_image.width
    canvas_height = preds_image.height + bar_gap + ground_truth_image.height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    canvas.paste(preds_image, (0, 0))
    canvas.paste(ground_truth_image, (0, preds_image.height + bar_gap))
    return canvas
