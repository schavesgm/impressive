"""Module containing functionality to caption images."""

from collections.abc import Callable
from typing import Protocol

import torch
from PIL.Image import Image

__all__ = ["Captioner", "Generator", "Processor", "build_captioner"]


class Processor[**K](Protocol):
    """Protocol for types allowing processing ``Images`` to produce some input tensor."""

    def __call__(
        self, image: Image | torch.Tensor, *args: K.args, **kwargs: K.kwargs
    ) -> torch.Tensor:
        """Return a processed version of the input ``Image`` object."""

    def batch_decode(self, inputs: torch.Tensor, *args: K.args, **kwargs: K.kwargs) -> str:
        """Return the decoded version of the ``Model`` output."""


class Generator[**K](Protocol):
    """Protocol for types allowing inferring captions from image tensors."""

    def generate(self, inputs: torch.Tensor, *args: K.args, **kwargs: K.kwargs) -> torch.Tensor:
        """Return some generated data from the input tensor."""


# Protocol for a function that produces captions.
type Captioner = Callable[[Image], list[str]]


def build_captioner(
    processor: Processor, generator: Generator, num_beams: int, num_sequences: int
) -> Captioner:
    """Return a ``Captioner`` function to caption ``Image`` objects.

    Args:
        processor (Processor): ``Processor`` type allowing pre-processing images.
        generator (Generator): ``Generator`` type allowing generating data from input.
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
