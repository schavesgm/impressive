"""Module containing functionality to caption images."""

import base64
from collections.abc import Callable
from enum import Enum
from io import BytesIO
from itertools import batched
from typing import NamedTuple, Protocol

import torch
from PIL.Image import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

__all__ = [
    "Captioner",
    "CaptionedImage",
    "PredefinedModels",
    "Inferer",
    "Processor",
    "build_captioner",
]


class CaptionedImage(NamedTuple):
    """Container for a captioned image."""

    image: Image
    captions: list[str]

    def as_base64(self) -> str:
        """Return the ``Image`` object as a ``base64`` string."""
        buffer = BytesIO()
        self.image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class Processor[**K](Protocol):
    """Protocol for types allowing processing ``Images`` to produce some input tensor."""

    def __call__(
        self, image: Image | torch.Tensor, *args: K.args, **kwargs: K.kwargs
    ) -> torch.Tensor:
        """Return a processed version of the input ``Image`` object."""

    def batch_decode(self, inputs: torch.Tensor, *args: K.args, **kwargs: K.kwargs) -> str:
        """Return the decoded version of the ``Model`` output."""


class Inferer[**K](Protocol):
    """Protocol for types allowing inferring captions from image tensors."""

    def generate(self, inputs: torch.Tensor, *args: K.args, **kwargs: K.kwargs) -> torch.Tensor:
        """Return some generated data from the input tensor."""


# Protocol for a function that produces captions.
type Captioner = Callable[[list[Image]], list[CaptionedImage]]


def build_captioner(
    processor: Processor,
    generator: Inferer,
    num_beams: int,
    num_sequences: int,
    batch_size: int,
    device: str,
) -> Captioner:
    """Return a ``Captioner`` function to caption ``Image`` objects.

    Args:
        processor (Processor): ``Processor`` type allowing pre-processing images.
        generator (Inferer): ``Inferer`` type allowing generating data from input.
        num_beams (int): Number of beams to use in the data generation.
        num_sequences (int): Number of sequences to generate.
        batch_size (int): Number of entries to process in parallel.
        device (str): Device where the model should be run.

    Returns:
        Captioner: Function to caption some ``Image`` objects.

    Raises:
        ValueError: If ``device`` is not ``"cuda"`` or ``"cpu"``.
    """
    config = {"num_beams": num_beams, "num_return_sequences": num_sequences, "max_new_tokens": 250}
    if device not in ("cuda", "cpu"):
        raise ValueError("Unrecognised device. Must be either 'cuda' or 'cpu'.")
    generator = generator.to(device)

    def _captioner(images: list[Image]) -> list[CaptionedImage]:
        """Return some generated captions for a collection of ``Image`` objects."""
        captioned_images: list[CaptionedImage] = []
        for image_batch in batched(images, batch_size):
            pixel_values = processor(images=image_batch, return_tensors="pt").pixel_values
            generated_ids = generator.generate(pixel_values=pixel_values.to(device), **config)
            generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            results = [
                CaptionedImage(
                    image=image,
                    captions=generated_captions[idx * num_sequences : (idx + 1) * num_sequences],
                )
                for idx, image in enumerate(image_batch)
            ]
            captioned_images.extend(results)
            del generated_ids, pixel_values
        return captioned_images

    return _captioner


class PredefinedModels(Enum):
    """Enumeration containing predefined captioning models."""

    SALESFORCE_BLIP: str = "Salesforce/blip-image-captioning-base"
    MICROSOFT_GIT: str = "microsoft/git-base-coco"

    @property
    def processor(self) -> Processor:
        """Return the ``Processor`` object to process input images."""
        match self:
            case PredefinedModels.SALESFORCE_BLIP:
                return BlipProcessor.from_pretrained(self.value)
            case PredefinedModels.MICROSOFT_GIT:
                return AutoProcessor.from_pretrained(self.value)

    @property
    def inferer(self) -> Processor:
        """Return the ``Inferer`` object to produce captions."""
        match self:
            case PredefinedModels.SALESFORCE_BLIP:
                return BlipForConditionalGeneration.from_pretrained(self.value)
            case PredefinedModels.MICROSOFT_GIT:
                return AutoModelForCausalLM.from_pretrained(self.value)
