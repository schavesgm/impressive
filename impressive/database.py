"""Module containing functionality to interface with the vector database."""

import base64
from collections.abc import Iterable
from enum import Enum
from io import BytesIO
from typing import NamedTuple

import ollama
import weaviate.classes as wvc
from PIL.Image import Image
from PIL.Image import open as create_image
from weaviate import Client
from weaviate.collections import Collection

__all__ = [
    "CaptionedImage",
    "RetrievedImage",
    "Similarity",
    "add_images",
    "get_image_collection",
    "request_images",
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


class Similarity(Enum):
    """Enumeration defining the similarity thresholds.

    Note:
        These thresholds are arbitrarily defined and should be fine-tuned for a production-ready
        application.
    """

    HIGH: float = 0.20
    MEDIUM: float = 0.50
    LOW: float = 1.20


class RetrievedImage(NamedTuple):
    """Container for a retrieved image from the collection."""

    image: Image
    distance: float

    @property
    def similarity(self) -> Similarity:
        """Return the similarity of the retrieved image to the requested caption."""
        if self.distance <= Similarity.HIGH.value:
            return Similarity.HIGH
        if self.distance <= Similarity.MEDIUM.value:
            return Similarity.MEDIUM
        return Similarity.LOW


def request_images(
    prompt: str, image_collection: Collection, ollama_model: str, num_images: int
) -> list[RetrievedImage]:
    """Request some images from the ``Collection`` object given some prompt and a model.

    Note:
        This function will filter out all requested images with values higher than
        ``Similarity.LOW``.

    Args:
        prompt (str): Prompt used to request the images from the collection.
        image_collection (Collection): Collection to query.
        ollama_model (str): ``ollama`` embedding model to use in the request.
        num_images (int): Number of nearest neighbours to retrieve.

    Returns:
        list[RetrievedImage]: Collection of requested images.
    """
    response = ollama.embeddings(model=ollama_model, prompt=prompt)
    results = image_collection.query.near_vector(
        near_vector=response["embedding"],
        limit=num_images,
        return_metadata=wvc.query.MetadataQuery(distance=True),
    )
    images = [
        RetrievedImage(
            image=_from_base64(result.properties["image"]), distance=result.metadata.distance
        )
        for result in results.objects
    ]
    return [image for image in images if image.distance <= Similarity.HIGH.value]


def add_images(
    image_collection: Collection, images: Iterable[CaptionedImage], ollama_model: str
) -> None:
    """Add some images to the database.

    Warning:
        This function has side-effects on the ``Collection``.

    Args:
        image_collection (Collection): Collection representing the image entries.
        images (Iterable[CaptionedImage]): Collection of ``CaptionedImage`` objects to add.
        ollama_model (str): Name of the ollama model to use when computing the embeddings.
    """
    with image_collection.batch.dynamic() as batch:
        for captioned_image in images:
            caption = ". ".join(captioned_image.captions)
            response = ollama.embeddings(model=ollama_model, prompt=caption)
            batch.add_object(
                properties={"image": captioned_image.as_base64(), "caption": caption},
                vector=response["embedding"],
            )


def get_image_collection(client: Client, quantise_vectors: bool = True) -> Collection:
    """Return the ``Image`` collection from the weaviate client.

    Warning:
        This function will create the collection if it is not present in the client.

    Note:
        The ``Image`` collection contains two properties: "caption" and "image". The "caption"
        property is some text that serves as the vector search key. The "image" is some ``base64``
        encoded image that contains the image.

    Args:
        client (Client): Weaviate server instance.
        quantise_vectors (bool): Flag selecting whether vectors should be quantised or not.

    Returns:
        Collection: ``Image`` collection.
    """
    match client.collections.exists("Image"):
        case True:
            return client.collections.get("Image")
        case False:
            return client.collections.create(
                name="Image",
                properties=[
                    wvc.config.Property(
                        name="image",
                        data_type=wvc.config.DataType.TEXT,
                        vectorize_property_name=False,
                    ),
                    wvc.config.Property(
                        name="caption",
                        data_type=wvc.config.DataType.TEXT,
                        vectorize_property_name=False,
                        tokenization=wvc.config.Tokenization.WHITESPACE,
                    ),
                ],
                **_get_vector_index_config(quantise_vectors),
            )


def _from_base64(encoding: str) -> Image:
    """Construct a ``Image`` object from a ``base64`` encoding.

    Args:
        encoding (str): ``base64`` encoding representing the image to create.

    Returns:
        Image: ``Image`` contained in the embedding.
    """
    return create_image(BytesIO(base64.b64decode(encoding)))


def _get_vector_index_config(quantise_vectors: bool) -> dict[str, wvc.config.Configure.VectorIndex]:
    """Return a dictionary with the vector index configuration.

    Args:
        quantise_vectors (bool): Flag denoting whether vectors should be quantised or not.

    Returns:
        dict[str, wcv.config.Configure.VectorIndex]: Dictionary containing the vector index
            strategy.
    """
    if not quantise_vectors:
        return {}
    return {
        "vector_index_config": wvc.config.Configure.VectorIndex.hnsw(
            quantizer=wvc.config.Configure.VectorIndex.Quantizer.bq()
        )
    }
