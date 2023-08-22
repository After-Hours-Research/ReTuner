import re
from typing import Union
from unstructured.partition.auto import partition
from unstructured.documents.elements import NarrativeText

from pathlib import Path

def process_document(
    document: Union[Path, str],
    min_chunk_size: int = 200,
    max_chunk_size: int = 500,
    end_on: list = [",", ".", "?", "!", "\n"],
):
    """
    Process a document into chunks of text.
    Input can either be a string (the document itself) or a path to a document.

    params:
        document: Union[Path, str]
            The document to process.
        min_chunk_size: int
            The minimum size of a chunk.
        max_chunk_size: int
            The maximum size of a chunk.
        end_on: list
            A list of characters to split a chunk on.
    """
    if isinstance(document, Path):
        chunks = partition(filename=str(document.resolve()))
        filtered_chunks = [c for c in chunks if isinstance(c, NarrativeText)]
        document = "\n".join([c.text for c in filtered_chunks])
    # Find all end points
    end_on_pattern = "|".join(map(re.escape, end_on))
    all_end_points = [match.end() for match in re.finditer(end_on_pattern, document)]
    
    # Initialize variables
    start = 0
    chunks = []
    
    # Loop through all end points
    for end in all_end_points:
        # If the end point is at least min_chunk_size characters away from the start,
        # add the chunk from start to end to the chunks list and update start to end
        if end - start >= min_chunk_size:
            chunk = document[start:end]
            # If the chunk is larger than max_chunk_size, split it into smaller chunks
            while len(chunk) > max_chunk_size:
                chunks.append(chunk[:max_chunk_size])
                chunk = chunk[max_chunk_size:]
            chunks.append(chunk)
            start = end
    
    # If there are any remaining characters after the last end point, add them to the chunks list
    if start < len(document):
        remaining = document[start:]
        # If the remaining characters are larger than max_chunk_size, split them into smaller chunks
        while len(remaining) > max_chunk_size:
            chunks.append(remaining[:max_chunk_size])
            remaining = remaining[max_chunk_size:]
        chunks.append(remaining)
    
    return chunks


