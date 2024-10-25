# Adapted from https://github.com/xor2k/npy-append-array/ by xor2k (MIT License)
import os
import struct
import tempfile
import threading
import warnings

import numpy
import numpy.compat

from numpy.lib import format as numpy_format

from io import BytesIO, SEEK_END, SEEK_SET
from math import prod, ceil

from types import TracebackType
from typing import Any, BinaryIO, Dict, Optional, Tuple, Type
from typing_extensions import Self

__all__ = ["AppendableNumpyArrayFile", "AppendableNumpyHeaderInfo"]

GROWTH_AXIS_MAX_DIGITS = 21

def _wrap_header(
    header: str,
    version: Tuple[int, int],
    header_len: Optional[int]=None
) -> bytes:
    """
    Takes a stringified header, and attaches the prefix and padding to it

    :param header: The header to wrap
    :param version: The version of the format
    :param header_len: If not None, pads the header to the specified value or
    :return: The wrapped header
    """
    assert version is not None
    fmt, encoding = numpy_format._header_size_info[version] # type: ignore[attr-defined]
    header_bytes = header.encode(encoding)
    hlen = len(header_bytes) + 1
    padlen = numpy_format.ARRAY_ALIGN - (
        (
            numpy_format.MAGIC_LEN + struct.calcsize(fmt) + hlen
        ) % numpy_format.ARRAY_ALIGN
    )

    try:
        packed_fmt = struct.pack(fmt, hlen + padlen)
        header_prefix = numpy_format.magic(*version) + packed_fmt # type: ignore[no-untyped-call]
    except struct.error:
        raise ValueError(f"Header length {hlen} too big for {version=}")

    # Pad the header with spaces and a final newline such that the magic
    # string, the header-length short and the header are aligned on a
    # ARRAY_ALIGN byte boundary. This supports memory mapping of dtypes
    # aligned up to ARRAY_ALIGN on systems like Linux where mmap()
    # offset must be page-aligned (i.e. the beginning of the file).
    header_bytes = header_prefix + header_bytes + b' '*padlen 
    
    if header_len is not None:
        actual_header_len = len(header_bytes) + 1
        if actual_header_len > header_len:
            msg = (
                "Header length {} too big for specified header "+
                "length {}, version={}"
            ).format(actual_header_len, header_len, version)
            raise ValueError(msg) from None
        header_bytes += b' '*(header_len - actual_header_len)
    
    return header_bytes + b'\n'

def _wrap_header_guess_version(
    header: str,
    header_len: Optional[int]=None
) -> bytes:
    """
    Like `_wrap_header`, but chooses an appropriate version given the contents

    :param header: The header to wrap
    :param header_len: If not None, pads the header to the specified value or
    :return: The wrapped header
    """
    try:
        return _wrap_header(header, (1, 0), header_len)
    except ValueError:
        pass

    try:
        ret = _wrap_header(header, (2, 0), header_len)
    except UnicodeEncodeError:
        pass
    else:
        warnings.warn(
            "Stored array in format 2.0. It can only be read by NumPy >= 1.9",
            UserWarning,
            stacklevel=2
        )
        return ret

    header_bytes = _wrap_header(header, (3, 0))
    warnings.warn(
        "Stored array in format 3.0. It can only be read by NumPy >= 1.17",
        UserWarning,
        stacklevel=2
    )
    return header_bytes

def _write_array_header(
    fp: BinaryIO,
    d: Dict[str, Any],
    version: Optional[Tuple[int, int]]=None,
    header_len: Optional[int]=None
) -> None:
    """
    Write the header for an array and returns the version used

    :param fp: file-like object
    :param d: dictionary containing the header data.
        This has the appropriate entries for writing its string representation
        to the header of the file.
    :param version: version of the format to use.
        None means use oldest that works. Providing an explicit version will
        raise a ValueError if the format does not allow saving this data.
    :param header_len: If not None, pads the header to the specified value or
        raises a ValueError if the header content is too big.
    """
    header_chunks = ["{"]
    for key, value in sorted(d.items()):
        # Need to use repr here, since we eval these when reading
        header_chunks.append("'%s': %s, " % (key, repr(value)))
    header_chunks.append("}")
    header = "".join(header_chunks)
    
    # Add some spare space so that the array header can be modified in-place
    # when changing the array size, e.g. when growing it by appending data at
    # the end. 
    shape = d['shape']
    header += " " * ((GROWTH_AXIS_MAX_DIGITS - len(repr(
        shape[-1 if d['fortran_order'] else 0]
    ))) if len(shape) > 0 else 0)
    
    if version is None:
        header_bytes = _wrap_header_guess_version(header, header_len)
    else:
        header_bytes = _wrap_header(header, version, header_len)

    fp.write(header_bytes)

def _write_array(
    fp: BinaryIO,
    array: numpy.ndarray[Any, Any],
    version: Optional[Tuple[int, int]]=None,
    allow_pickle: bool=True,
    pickle_kwargs: Optional[Dict[str, Any]]=None
) -> None:
    """
    Write an array to an NPY file, including a header.
    If the array is neither C-contiguous nor Fortran-contiguous AND the
    file_like object is not a real file object, this function will have to
    copy data in memory.
    :param fp: file-like object
        An open, writable file object, or similar object with a
        ``.write()`` method.
    :param array: ndarray
        The array to write to disk.
    version : (int, int) or None, optional
        The version number of the format. None means use the oldest
        supported version that is able to store the data.  Default: None
    allow_pickle : bool, optional
        Whether to allow writing pickled data. Default: True
    pickle_kwargs : dict, optional
        Additional keyword arguments to pass to pickle.dump, excluding
        'protocol'. These are only useful when pickling objects in object
        arrays on Python 3 to Python 2 compatible format.
    Raises
    ------
    ValueError
        If the array cannot be persisted. This includes the case of
        allow_pickle=False and array being an object array.
    Various other errors
        If the array contains Python objects as part of its dtype, the
        process of pickling them may raise various errors if the objects
        are not picklable.
    """
    numpy_format._check_version(version) # type: ignore[attr-defined]
    _write_array_header(fp, numpy_format.header_data_from_array_1_0(array), version) # type: ignore[no-untyped-call]

    if array.itemsize == 0:
        buffer_size = 0
    else:
        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffer_size = max(16 * 1024 ** 2 // array.itemsize, 1)

    if array.dtype.hasobject:
        # We contain Python objects so we cannot write out the data
        # directly.  Instead, we will pickle it out
        if not allow_pickle:
            raise ValueError("Object arrays cannot be saved when allow_pickle=False")
        if pickle_kwargs is None:
            pickle_kwargs = {}
        numpy.compat.pickle.dump(array, fp, protocol=3, **pickle_kwargs)
    elif array.flags.f_contiguous and not array.flags.c_contiguous:
        if numpy.compat.isfileobj(fp): # type: ignore[no-untyped-call]
            array.T.tofile(fp)
        else:
            for chunk in numpy.nditer(
                array,
                flags=["external_loop", "buffered", "zerosize_ok"],
                buffersize=buffer_size,
                order="F"
            ):
                fp.write(chunk.tobytes("C")) # type: ignore[attr-defined]
    else:
        if numpy.compat.isfileobj(fp): # type: ignore[no-untyped-call]
            array.tofile(fp)
        else:
            for chunk in numpy.nditer(
                array,
                flags=["external_loop", "buffered", "zerosize_ok"],
                buffersize=buffer_size,
                order="C"
            ):
                fp.write(chunk.tobytes("C")) # type: ignore[attr-defined]

class AppendableNumpyHeaderInfo:
    """
    A class for reading the header of a numpy file
    """
    shape: Tuple[int, ...]
    fortran_order: bool
    dtype: numpy.dtype[Any]
    header_size: int
    data_length: int
    is_appendable: bool
    needs_recovery: bool

    def __init__(self, fp: BinaryIO) -> None:
        """
        Reads the header of a numpy file

        :param fp: file-like object
        """
        version = numpy_format.read_magic(fp) # type: ignore[no-untyped-call]
        self.shape, self.fortran_order, self.dtype = numpy_format._read_array_header(fp, version) # type: ignore[attr-defined]
        self.header_size = fp.tell()

        new_header = BytesIO()
        _write_array_header(
            new_header,
            {
                "shape": self.shape,
                "fortran_order": self.fortran_order,
                "descr": numpy_format.dtype_to_descr(self.dtype) # type: ignore[no-untyped-call]
            }
        )
        self.new_header = new_header.getvalue()
        fp.seek(0, SEEK_END)

        self.data_length = fp.tell() - self.header_size
        self.is_appendable = len(self.new_header) <= self.header_size
        self.needs_recovery = not (
            self.dtype.hasobject or
            self.data_length == prod(self.shape) * self.dtype.itemsize
        )

    @classmethod
    def file_is_appendable(cls, filename: str) -> bool:
        """
        Checks if a numpy file is appendable

        :param filename: The path to the file
        :return: True if the file is appendable, False otherwise
        """
        with open(filename, mode="rb") as fp:
            return cls(fp).is_appendable

    @classmethod
    def file_needs_recovery(cls, filename: str) -> bool:
        """
        Checks if a numpy file needs recovery

        :param filename: The path to the file
        :return: True if the file needs recovery, False otherwise
        """
        with open(filename, mode="rb") as fp:
            return cls(fp).needs_recovery

    @classmethod
    def ensure_appendable(
        cls,
        filename: str,
        in_place: bool=False
    ) -> None:
        """
        Ensures that a numpy file is appendable

        :param filename: The path to the file
        :param in_place: Whether to modify the file in-place
        """
        fp2: Optional[BinaryIO] = None

        with open(filename, mode="rb+") as fp:
            header_info = cls(fp)
            if header_info.is_appendable:
                return

            new_header_size = len(header_info.new_header)
            new_header, header_size = header_info.new_header, header_info.header_size
            data_length = header_info.data_length

            # Set buffer size to 16 MiB to hide the Python loop overhead, see
            # https://github.com/numpy/numpy/blob/main/numpy/lib/format.py
            buffer_size = min(16 * 1024 ** 2, data_length)
            buffer_count = int(ceil(data_length / buffer_size))

            if in_place:
                for i in reversed(range(buffer_count)):
                    offset = i * buffer_size
                    fp.seek(header_size + offset, SEEK_SET)
                    content = fp.read(buffer_size)
                    fp.seek(new_header_size + offset, SEEK_SET)
                    fp.write(content)

                fp.seek(0, SEEK_SET)
                fp.write(new_header)
                return

            dirname, basename = os.path.split(fp.name)
            output_tempfile = tempfile.NamedTemporaryFile(
                prefix=basename,
                dir=dirname,
                delete=False
            )
            fp2 = open(output_tempfile.name, mode="wb+")
            fp2.write(new_header)

            fp.seek(header_size, SEEK_SET)
            for _ in range(buffer_count):
                fp2.write(fp.read(buffer_size))

        assert fp2 is not None
        fp2.close()
        os.replace(fp2.name, fp.name)

    @classmethod
    def recover(
        cls,
        filename: str,
        zerofill_incomplete: bool=False
    ) -> None:
        """
        Recovers a numpy file

        :param filename: The path to the file
        :param zerofill_incomplete: Whether to zero-fill incomplete data
        """
        with open(filename, mode="rb+") as fp:
            header_info = cls(fp)
            shape, fortran_order, dtype = header_info.shape, header_info.fortran_order, header_info.dtype
            header_size, data_length = header_info.header_size, header_info.data_length

            if not header_info.needs_recovery:
                return

            assert header_info.is_appendable, "header not appendable, call ensure_appendable first"

            append_axis_itemsize = prod(
                shape[slice(None, None, -1 if fortran_order else 1)][1:]
            ) * dtype.itemsize

            trailing_bytes = data_length % append_axis_itemsize

            if trailing_bytes != 0:
                if zerofill_incomplete is True:
                    zero_bytes_to_append = append_axis_itemsize - trailing_bytes
                    fp.write(b'\0'*(zero_bytes_to_append))
                    data_length += zero_bytes_to_append
                else:
                    fp.truncate(header_size + data_length - trailing_bytes)
                    data_length -= trailing_bytes

            new_shape = list(shape)
            new_shape[-1 if fortran_order else 0] = data_length // append_axis_itemsize
            fp.seek(0, SEEK_SET)
            _write_array_header(
                fp,
                {
                    "shape": tuple(new_shape),
                    "fortran_order": fortran_order,
                    "descr": numpy_format.dtype_to_descr(dtype) # type: ignore[no-untyped-call]
                },
                header_len=header_size
            )

class AppendableNumpyArrayFile:
    """
    Class for appending numpy arrays to a file.

    >>> import numpy
    >>> import tempfile
    >>> tf = tempfile.NamedTemporaryFile()
    >>> with AppendableNumpyArrayFile(tf.name) as f:
    ...     f.append(numpy.array([1, 2, 3]))
    ...     f.append(numpy.array([4, 5, 6]))
    >>> numpy.load(tf.name)
    array([1, 2, 3, 4, 5, 6])
    """
    lock: threading.Lock
    initialized: bool
    fp: Optional[BinaryIO] = None
    header_length: Optional[int] = None

    def __init__(
        self,
        filename: str,
        delete_if_exists: bool=False,
        rewrite_header_on_append: bool=True
    ) -> None:
        self.filename = filename
        self.rewrite_header_on_append = rewrite_header_on_append
        self.lock = threading.Lock()
        self.initialized = False

        if os.path.exists(filename):
            if os.path.getsize(filename) == 0 or delete_if_exists:
                os.unlink(filename)
            else:
                self.initialize_file()

    def initialize_file(self) -> None:
        """
        Initializes the file
        """
        self.fp = open(self.filename, "rb+")
        header_info = AppendableNumpyHeaderInfo(self.fp)
        (
            self.shape,
            self.fortran_order,
            self.dtype,
            self.header_length
        ) = (
            header_info.shape,
            header_info.fortran_order,
            header_info.dtype,
            header_info.header_size
        )

        if self.dtype.hasobject:
            raise ValueError("Object arrays cannot be appended to")

        if not header_info.is_appendable:
            raise ValueError(" ".join([
                f"Header of {self.filename} not appendable.",
                "Call `AppendableNumpyArrayFileHeaderInfo.ensure_appendable`"
            ]))

        if header_info.needs_recovery:
            raise ValueError(" ".join([
                f"Cannot append to {self.filename}, needs recovery.",
                "Call `AppendableNumpyArrayFileHeaderInfo.recover`"
            ]))

        self.initialized = True

    def _write_array_header(self) -> None:
        """
        Writes the array header
        """
        if not self.fp:
            return
        self.fp.seek(0, SEEK_SET)
        _write_array_header(
            self.fp,
            {
                "shape": self.shape,
                "fortran_order": self.fortran_order,
                "descr": numpy_format.dtype_to_descr(self.dtype) # type: ignore[no-untyped-call]
            },
            header_len = self.header_length
        )

    def update_header(self) -> None:
        """
        Updates the header
        """
        with self.lock:
            self._write_array_header()

    def append(self, arr: numpy.ndarray[Any, Any]) -> None:
        """
        Appends an array to the file

        :param arr: The array to append
        """
        with self.lock:
            if not self.initialized:
                # If the file never existed or was deleted, just write
                # The current array to it
                with open(self.filename, "wb") as fp:
                    _write_array(fp, arr)
                return self.initialize_file()

            fortran_coeff = -1 if self.fortran_order else 1
            current_shape = self.shape[::fortran_coeff][1:][::fortran_coeff]
            appending_shape = arr.shape[::fortran_coeff][1:][::fortran_coeff]

            if current_shape != appending_shape:
                raise ValueError(f"Shapes {current_shape} and {appending_shape} do not match")

            # Seek to the end of the file
            assert self.fp is not None
            self.fp.seek(0, SEEK_END)

            # Write the array to the file
            arr.astype(
                self.dtype,
                copy=False
            ).flatten(
                order='F' if self.fortran_order else 'C'
            ).tofile(self.fp)

            if self.fortran_order:
                self.shape = (*self.shape[:-1], self.shape[-1] + arr.shape[-1])
            else:
                self.shape = (self.shape[0] + arr.shape[0], *self.shape[1:])

            if self.rewrite_header_on_append:
                self._write_array_header()

    def close(self) -> None:
        """
        Closes the file
        """
        with self.lock:
            if self.initialized:
                if not self.rewrite_header_on_append:
                    self._write_array_header()

                assert self.fp is not None
                self.fp.close()
                self.initialized = False

    def __del__(self) -> None:
        """
        Destructor
        """
        self.close()

    def __enter__(self) -> Self:
        """
        Context manager enter
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]]=None,
        exc_val: Optional[BaseException]=None,
        exc_tb: Optional[TracebackType]=None
    ) -> None:
        """
        Context manager exit
        """
        self.__del__()
