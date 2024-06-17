"""Entry point of the `impressive` application."""

import streamlit as st
import weaviate
from weaviate.collections import Collection

from impressive.database import Similarity, get_image_collection, request_images

# Model to use in the vector database search
EMBEDDING_MODEL: str = "nomic-embed-text"

# Number of images to retrieve
NUM_IMAGES: int = 5

# Text to write as header to the webpage
DESCRIPTION: str = """
<div style='text-align: justify;'>
    Impressive is a search tool to request impressionist paintings using natural language
    processing. The system will retrieve paintings from a vector database containing
    impressionist paintings. The captions used in the search were automatically generated using
    an image-captioning model. When a query is sent to the database, a maximum of 5 images will
    be requested. The images are ordered by their similarity to the queried prompt.
</div>
"""


def add_blank_space(margin: str) -> None:
    """Add some blank space to a ``streamlit`` application."""
    st.write(f"<div style='margin: {margin};'>", unsafe_allow_html=True)
    st.write("")
    st.write("</div>", unsafe_allow_html=True)


def _get_similarity_caption(similarity: Similarity) -> str:
    """Return a stylised caption for each similarity level.

    Args:
        similarity (Similarity): Similarity level.

    Returns:
        str: Caption of each similarity level.
    """
    match similarity:
        case Similarity.HIGH:
            return "🚀 High similarity"
        case Similarity.MEDIUM:
            return "🤩 Medium similarity"
        case Similarity.LOW:
            return "👎 Low similarity"


def draw_images(prompt: str, collection: Collection) -> None:
    """Request some images with a prompt and draw them into the application.

    Args:
        prompt (str): Prompt to use in the image search.
        collection (Collection): Image database.
    """
    results = request_images(prompt, collection, EMBEDDING_MODEL, num_images=NUM_IMAGES)
    if len(results) == 0:
        st.subheader("⚠️ No paintings retrieved. Try another prompt.")
        return
    st.subheader("🖼️ Retrieved paintings")
    columns = st.columns(len(results))
    for column, result in zip(columns, results, strict=True):
        with column:
            st.image(
                result.image,
                caption=_get_similarity_caption(result.similarity),
                use_column_width=True,
            )


def render_search_engine(image_collection: Collection) -> str:
    """Render the search engine in the web-page.

    Args:
        image_collection (Collection): Collection containing all images to use in the search.
    """
    st.write("Need inspiration?")
    column_1, column_2, column_3, _ = st.columns([1.0, 1.0, 1.0, 3.0], gap="small")
    example_prompt: str = ""
    with column_1:
        if st.button("🌳 Gardens"):
            example_prompt = "paintings of a garden"
    with column_2:
        if st.button("🌻 Flowers"):
            example_prompt = "paintings of flowers"
    with column_3:
        if st.button("🐎 Horses"):
            example_prompt = "paintings of horses"

    user_input = st.text_input("Request some paintings:")
    if st.button("Request"):
        if user_input:
            draw_images(user_input, image_collection)
        else:
            st.subheader("⚠️ Please enter some text before submitting.")

    if example_prompt:
        draw_images(example_prompt, image_collection)


def main() -> None:
    """Entry point of the script."""
    client = weaviate.connect_to_local()
    image_collection = get_image_collection(client, quantise_vectors=True)

    st.title("🎨 Impressive")
    add_blank_space(margin="1em")
    st.markdown(DESCRIPTION, unsafe_allow_html=True)
    add_blank_space(margin="1em")

    if len(image_collection) == 0:
        st.subheader("⚠️ The database does not contain any images.")
    else:
        render_search_engine(image_collection)


if __name__ == "__main__":
    main()
