"""Module containing functionality to caption images."""

from collections.abc import Callable
from enum import Enum
from typing import Protocol

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
    "PredefinedModels",
    "Inferer",
    "Processor",
    "build_captioner",
]


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
type Captioner = Callable[[Image], list[str]]


def build_captioner(
    processor: Processor, generator: Inferer, num_beams: int, num_sequences: int
) -> Captioner:
    """Return a ``Captioner`` function to caption ``Image`` objects.

    Args:
        processor (Processor): ``Processor`` type allowing pre-processing images.
        generator (Inferer): ``Inferer`` type allowing generating data from input.
        num_beams (int): Number of beams to use in the data generation.
        num_sequences (int): Number of sequences to generate.

    Returns:
        Captioner: Function to caption some ``Image`` objects.
    """
    config = {"num_beams": num_beams, "num_return_sequences": num_sequences, "max_new_tokens": 250}
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    generator = generator.to(device)

    def _captioner(image: Image) -> list[str]:
        """Return some generated captions for the input ``Image`` object."""
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = generator.generate(pixel_values=pixel_values, **config)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)

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
