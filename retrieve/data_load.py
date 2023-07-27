import re
from unstructured.partition.auto import partition
from unstructured.documents.elements import NarrativeText

from pathlib import Path

def load_document(
    document_path: Path,
    min_chunk_size: int = 200,
    max_chunk_size: int = 500,
    end_on: list = [",", ".", "?", "!", "\n"],
):
    chunks = partition(filename=str(document_path.resolve()))
    filtered_chunks = [c for c in chunks if isinstance(c, NarrativeText)]
    joined = "\n".join([c.text for c in filtered_chunks])
    end_on_pattern = "|".join(map(re.escape, end_on))
    
    # Find all end points
    all_end_points = [match.end() for match in re.finditer(end_on_pattern, joined)]
    
    # Initialize variables
    start = 0
    chunks = []
    
    # Loop through all end points
    for end in all_end_points:
        # If the end point is at least min_chunk_size characters away from the start,
        # add the chunk from start to end to the chunks list and update start to end
        if end - start >= min_chunk_size:
            chunk = joined[start:end]
            # If the chunk is larger than max_chunk_size, split it into smaller chunks
            while len(chunk) > max_chunk_size:
                chunks.append(chunk[:max_chunk_size])
                chunk = chunk[max_chunk_size:]
            chunks.append(chunk)
            start = end
    
    # If there are any remaining characters after the last end point, add them to the chunks list
    if start < len(joined):
        remaining = joined[start:]
        # If the remaining characters are larger than max_chunk_size, split them into smaller chunks
        while len(remaining) > max_chunk_size:
            chunks.append(remaining[:max_chunk_size])
            remaining = remaining[max_chunk_size:]
        chunks.append(remaining)
    
    return chunks


