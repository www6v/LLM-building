# https://python.langchain.com/docs/how_to/recursive_text_splitter/
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load example document
# with open("state_of_the_union.txt") as f:
#     state_of_the_union = f.read()

with open("zhongwen.txt", encoding='UTF-8') as f:
    state_of_the_union = f.read()
    

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=True,

    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200B",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    # Existing args
)



texts = text_splitter.create_documents([state_of_the_union])
# print(texts[0])
# print(texts[1])

for text in texts:
    print(text)