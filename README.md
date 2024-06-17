# 🎨 Impressive

`Impressive` is a small application leveraging vector embeddings to produce a natural language
search engine over images. `Impressive` depends on a small Python package called `impressive`, which
contains functionality to simplify captioning images using image-to-text models such as
[BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base), as well as functionality to
create and interface with a [`weaviate`](https://weaviate.io/developers/weaviate/client-libraries)
vector database instance.

In its current format, `Impressive` is designed to work in a local environment. This ensures that no
API keys are required to generate the database vectors. In order to install the required
`impressive` package used in the application and its dependencies, the following steps must be
completed: (1) create a virtual environment in your preferred distribution (
[`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html),
[`venv`](https://docs.python.org/3/library/venv.html)) with `Python >= 3.12`; (2) activate the newly
created virtual environment; (3) install the package and its dependencies using `pip`,

```bash
pip install "." --index-url https://download.pytorch.org/whl/cu121
```

## Running the application

`Impressive` is served as small [`streamlit`](https://streamlit.io/) application. The entry point of
the application is `app.py`. However, in order to run the application, we must first initialise a
local `weaviate` server. This can be easily done using [`Docker`](https://www.docker.com/). To
download and instantiate the local `weaviate` server, run

```bash
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.25.4
```

> ⚠ Make sure to pull the latest version of the `weaviate` client from the `weaviate` 
[documentation](https://weaviate.io/developers/weaviate/installation/docker-compose).

Once the `weaviate` server is up and running, we need to create the database that will be used in
the search application. To do so, a guided [Jupyter notebook](./scripts/create_database.ipynb) is
provided. The notebook pulls an example [dataset of impressionist paintings]
(https://huggingface.co/datasets/chashaotm/impressionist_paintings), automatically captions a subset
of all images in the dataset, creates a `weaviate`
[`Collection`](https://weaviate.io/developers/weaviate/manage-data/collections) called `Image` to
store the captioned images, and stores the images. To run the notebook, the 
[`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) `ollama` model must be installed:

```bash
ollama pull nomic-embed-text
```

The notebook provides a simple example, but given the flexibility of the `impressive` package,
extending the application to other datasets or captioning mechanisms should be simple.

> ⚠ Make sure to run the notebook while the `weaviate` local server is running.

Once the image database is created on the local `weaviate` server, we can spawn the application. To
do so, run

```bash
streamlit run app.py
```

You should now be able to query impressionist paintings in your browser using natural language
processing! 🚀🚀

## Remarks

The `Impressive` application is a small proof of concept exploring the use of vector databases and
the current space of AI models to caption images and embed text. It does not represent a deployable
or complete application. It is also not fine-tuned for production. As a result, it might contain
false positives (as the parameters of the distance search are not tuned properly), it might not use
the most performant models available, and it might not make use of the latest trends in vector
databases.
