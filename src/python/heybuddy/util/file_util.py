import io
import os
import requests

from typing import Optional, Callable, Union, BinaryIO, Iterator
from contextlib import contextmanager

from heybuddy.util.log_util import logger

__all__ = [
    "check_download_file",
    "check_download_file_to_dir",
    "get_file_name_from_url",
    "get_domain_from_url",
    "file_is_downloaded",
    "file_is_downloaded_to_dir",
    "check_remove_interrupted_download",
    "retrieve_uri",
]

def file_matches_sha256_sum(
    path: str,
    sha256_sum: str,
    chunk_size: int=4096
) -> bool:
    """
    Checks if a file matches a sha256_sum.

    >>> import tempfile
    >>> tempfile = tempfile.NamedTemporaryFile()
    >>> open(tempfile.name, "wb").write(b"test")
    4
    >>> file_matches_sha256_sum(tempfile.name, "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")
    True

    :param path: The path to the file
    :param sha256_sum: The sha256_sum to check against
    :param chunk_size: The size of the chunks to read. Defaults to 4096 bytes.
    :return: True if the sha256_sum matches, False otherwise
    """
    import hashlib
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest() == sha256_sum

def get_domain_from_url(url: str) -> str:
    """
    Gets a domain from a URL.

    >>> get_domain_from_url("http://example.com")
    'example.com'
    >>> get_domain_from_url("http://example.com/file.txt")
    'example.com'
    >>> get_domain_from_url("https://test.example.com/")
    'test.example.com'

    :param url: The URL to get the domain from
    :return: The domain
    """
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    return parsed_url.netloc

def get_file_name_from_url(url: str) -> str:
    """
    Gets a filename from a URL.

    Checks for common ways to specify filenames in URLs,
    before falling back to the last part of the URL.

    >>> get_file_name_from_url("http://example.com/file.txt")
    'file.txt'
    >>> get_file_name_from_url("http://example.com/file.txt?filename=example.txt")
    'example.txt'
    >>> get_file_name_from_url("http://example.com/file.txt?response-content-disposition=attachment; filename=example.txt")
    'example.txt'

    :param url: The URL to get the filename from
    :return: The filename
    """
    from urllib.parse import urlparse, parse_qs
    parsed_url = urlparse(url)
    parsed_qs = parse_qs(parsed_url.query)
    if "filename" in parsed_qs:
        return parsed_qs["filename"][0]
    elif "response-content-disposition" in parsed_qs:
        disposition_parts = parsed_qs["response-content-disposition"][0].split(";")
        for part in disposition_parts:
            part_data = part.strip("'\" ").split("=")
            if len(part_data) < 2:
                continue
            part_key, part_value = part_data[0], "=".join(part_data[1:])
            if part_key == "filename":
                return part_value.strip("'\" ")
    return os.path.basename(url.split("?")[0])

def check_remove_interrupted_download(
    remote_url: str,
    target: str,
    authorization: Optional[str]=None,
) -> bool:
    """
    Checks if a file exists and is an interrupted download by comparing the
    size to the remote URL's size from a HEAD request. If it is, removes the file.

    :param remote_url: The URL to check against
    :param target: The file to check
    :param authorization: The authorization header to use. Defaults to None.
    :return: True if the file was removed, False otherwise
    """
    if os.path.exists(target):
        headers = {}
        if authorization is not None:
            headers["Authorization"] = authorization
        elif "huggingface.co" in remote_url or "hf.co" in remote_url:
            hf_token = os.getenv("HF_TOKEN", None)
            if hf_token is not None:
                headers["Authorization"] = f"Bearer {hf_token}"

        head = requests.head(remote_url, headers=headers, allow_redirects=True)
        head.raise_for_status()
        expected_length = head.headers.get("Content-Length", None)
        actual_length = os.path.getsize(target)
        if expected_length and actual_length != int(expected_length):
            logger.info(
                f"File at {target} looks like an interrupted download, or the remote resource has changed - expected a size of {expected_length} bytes but got {actual_length} instead. Removing."
            )
            os.remove(target)
            return True
    return False

def check_remove_non_matching_sha256_sum(
    sha256_sum: str,
    target: str,
    chunk_size: int=4096
) -> bool:
    """
    Checks if a file exists and does not match a sha256_sum.
    If it does not, removes the file.

    :param sha256_sum: The sha256_sum to check against
    :param target: The file to check
    :param chunk_size: The size of the chunks to read. Defaults to 4096 bytes.
    :return: True if the file was removed, False otherwise
    """
    if os.path.exists(target):
        if not file_matches_sha256_sum(target, sha256_sum, chunk_size=chunk_size):
            logger.info(f"File at {target} does not match the expected sha256_sum. Removing.")
            os.remove(target)
            return True
    return False

def file_is_downloaded(
    remote_url: str,
    target: str,
    check_size: bool=True,
    sha256_sum: Optional[str]=None,
) -> bool:
    """
    Checks if a file exists and matches the remote URL's size.

    :param remote_url: The URL to check against
    :param target: The file to check
    :param check_size: Whether to check the size. Defaults to True.
    :param sha256_sum: The sha256_sum to check against. Defaults to None.
    :return: True if the file exists and matches, False otherwise
    """
    if sha256_sum is not None and check_remove_non_matching_sha256_sum(sha256_sum, target):
        return False
    if check_size and check_remove_interrupted_download(remote_url, target):
        return False
    return os.path.exists(target)

def file_is_downloaded_to_dir(
    remote_url: str,
    local_dir: str,
    file_name: Optional[str]=None,
    check_size: bool=True,
    sha256_sum: Optional[str]=None,
) -> bool:
    """
    Checks if a file exists in a directory based on a remote path.
    If it does, checks the size and matches against the remote URL.

    :param remote_url: The URL to check against
    :param local_dir: The directory to check in
    :param file_name: The filename to check. Defaults to None.
    :param check_size: Whether to check the size. Defaults to True.
    :param sha256_sum: The sha256_sum to check against. Defaults to None.
    :return: True if the file exists and matches, False otherwise
    """
    if file_name is None:
        file_name = get_file_name_from_url(remote_url)
    local_path = os.path.join(local_dir, file_name)
    return file_is_downloaded(
        remote_url,
        local_path,
        check_size=check_size,
        sha256_sum=sha256_sum
    )

def check_download_file(
    remote_url: str,
    target: Union[str, BinaryIO],
    chunk_size: int=8192,
    check_size: bool=True,
    use_tqdm: bool=False,
    resume_size: int = 0,
    progress_callback: Optional[Callable[[int, int], None]]=None,
    authorization: Optional[str]=None,
    sha256_sum: Optional[str]=None,
) -> None:
    """
    Checks if a file exists.
    If it does, checks the size and matches against the remote URL.
    If it doesn't, or the size doesn't match, download it.

    :param remote_url: The URL to check against
    :param target: The file to check or write to. If a string, writes to a file. If a file handle, writes to that. If a file handle, writes to that.
    :param chunk_size: The size of the chunks to read. Defaults to 8192 bytes.
    :param check_size: Whether to check the size. Defaults to True.
    :param resume_size: The size to resume from. Defaults to 0.
    :param progress_callback: The callback to call with progress. Defaults to None.
    :param authorization: The authorization header to use. Defaults to None.
    :param sha256_sum: The sha256_sum to check against. Defaults to None.
    """
    try:
        headers = {}
        if authorization is not None:
            headers["Authorization"] = authorization
        elif "huggingface.co" in remote_url or "hf.co" in remote_url:
            hf_token = os.getenv("HF_TOKEN", None)
            if hf_token is not None:
                headers["Authorization"] = f"Bearer {hf_token}"

        if isinstance(target, str) and check_size and resume_size <= 0:
            # Remove interrupted downloads if we aren't resuming it
            check_remove_interrupted_download(remote_url, target, authorization=authorization)
        if isinstance(target, str) and sha256_sum is not None and resume_size <= 0:
            # Remove non-matching downloads if we aren't resuming it
            check_remove_non_matching_sha256_sum(sha256_sum, target)

        if resume_size is not None:
            headers["Range"] = f"bytes={resume_size:d}-"

        if not isinstance(target, str) or not os.path.exists(target):
            @contextmanager
            def get_write_handle() -> Iterator[BinaryIO]:
                if isinstance(target, str):
                    with open(target, "wb") as handle:
                        yield handle
                else:
                    yield target

            logger.info(f"Downloading file from {remote_url}. Will write to {target}")
            response = requests.get(remote_url, allow_redirects=True, stream=True, headers=headers)
            response.raise_for_status()
            content_length: Optional[int] = response.headers.get("Content-Length", None) # type: ignore[assignment]
            if content_length is not None:
                content_length = int(content_length)

            if use_tqdm:
                import tqdm
                pbar = tqdm.tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {os.path.basename(remote_url)} from {get_domain_from_url(remote_url)}",
                    mininterval=1.0,
                )

                original_progress_callback = progress_callback

                def tqdm_progress_callback(written_bytes: int, total_bytes: int) -> None:
                    pbar.update(written_bytes - pbar.n)
                    if original_progress_callback is not None:
                        original_progress_callback(written_bytes, total_bytes)

                progress_callback = tqdm_progress_callback

            with get_write_handle() as fh:
                written_bytes = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    if progress_callback is not None and content_length is not None:
                        written_bytes = min(written_bytes + chunk_size, content_length)
                        progress_callback(written_bytes, content_length)
    except Exception as e:
        logger.error(f"Received an error while downloading file from {remote_url}: {e}")
        if isinstance(target, str) and os.path.exists(target):
            logger.debug(f"File exists on-disk at {target}, falling back to that.")
            return
        raise

def check_download_file_to_dir(
    remote_url: str,
    local_dir: str,
    use_tqdm: bool=False,
    file_name: Optional[str]=None,
    chunk_size: int=8192,
    check_size: bool=True,
    progress_callback: Optional[Callable[[int, int], None]]=None,
    authorization: Optional[str]=None,
    sha256_sum: Optional[str]=None,
) -> str:
    """
    Checks if a file exists in a directory based on a remote path.
    If it does, checks the size and matches against the remote URL.
    If it doesn't, or the size doesn't match, download it.

    :param remote_url: The URL to check against
    :param local_dir: The directory to check in
    :param file_name: The filename to check. Defaults to None.
    :param chunk_size: The size of the chunks to read. Defaults to 8192 bytes.
    :param check_size: Whether to check the size. Defaults to True.
    :param progress_callback: The callback to call with progress. Defaults to None.
    :param authorization: The authorization header to use. Defaults to None.
    :param sha256_sum: The sha256 sum to check against. Defaults to None.
    :return: The local path downloaded to
    """
    if file_name is None:
        file_name = get_file_name_from_url(remote_url)

    local_path = os.path.join(local_dir, file_name)
    check_download_file(
        remote_url,
        local_path,
        use_tqdm=use_tqdm,
        chunk_size=chunk_size,
        check_size=check_size,
        progress_callback=progress_callback,
        authorization=authorization,
        sha256_sum=sha256_sum
    )
    return local_path

def retrieve_uri(uri: str, chunk_size: int=8192) -> BinaryIO:
    """
    Retrieves a URI as a stream of bytes
    When fruition is available, uses that, which supports more protocols.
    When not available, supports http/s and files.

    This method should be used for data that never needs to be cached.
    If you need to download the file once and cache it, use check_download_file instead.

    >>> import os
    >>> import tempfile
    >>> retrieve_uri("http://example.com").read().decode("utf-8")[:15]
    '<!doctype html>'
    >>> tempfile = tempfile.NamedTemporaryFile()
    >>> open(tempfile.name, "wb").write(b"test")
    4
    >>> retrieve_uri(f"file://{tempfile.name}").read().decode("utf-8")
    'test'
    >>> retrieve_uri(tempfile.name).read().decode("utf-8")
    'test'

    :param uri: The URI to retrieve
    :param chunk_size: The size of the chunks to read. Defaults to 8192 bytes.
    :return: A stream of bytes
    """
    try:
        from fruition.resources.retriever import RetrieverIO # type: ignore[import-not-found,unused-ignore]
        return RetrieverIO(uri) # type: ignore[no-any-return]
    except ImportError:
        if uri.startswith("http"):
            from requests import get
            return io.BytesIO(get(uri, stream=True).content)
        else:
            if uri.startswith("file://"):
                uri = uri[7:]
            return open(uri, "rb")
