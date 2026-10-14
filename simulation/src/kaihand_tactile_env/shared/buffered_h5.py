"""Bounded, lossless append buffering for existing recorder HDF5 operations.

Independent general-purpose copy of the validated USB-private adapter. USB keeps
its implementation and source identity unchanged. Samples are copied immediately,
camera streams remain immediate, and flush/close drain all completed appends.
This is a narrow recorder adapter, not a replacement for the complete h5py API.
"""

from __future__ import annotations

import h5py
import numpy as np


class BufferedDataset:
  def __init__(self, dataset, rows):
    self._dataset = dataset
    self._rows = rows
    self._size, *tail = dataset.shape
    self._tail = tuple(tail)
    self._dtype = dataset.dtype
    self._pending = []
    self._count = 0
    self._reservation = None
    self.flush_count = 0

  @property
  def shape(self):
    return (self._size, *self._tail)

  def __getattr__(self, name):
    return getattr(self._dataset, name)

  def resize(self, size, axis=0):
    if axis != 0 or not isinstance(size, (int, np.integer)) or size < self._size:
      raise ValueError("buffer supports append-only resize on axis 0")
    if self._reservation is not None:
      raise RuntimeError("previous append reservation has not been written")
    if size != self._size:
      self._reservation = (self._size, int(size))
      self._size = int(size)

  def __setitem__(self, key, value):
    if self._reservation is None:
      # Terminal bookkeeping may edit an existing phase/index, never add a row.
      self.flush()
      self._dataset[key] = value
      return
    start, end = self._reservation
    single = isinstance(key, (int, np.integer)) and key == start and end == start + 1
    many = (
      isinstance(key, slice)
      and key.start == start
      and key.stop in (None, end)
      and key.step in (None, 1)
    )
    if not (single or many):
      raise ValueError("append must fill exactly its reserved rows")
    block = np.empty((end - start, *self._tail), dtype=self._dtype)
    block[...] = value
    self._pending.append(block)
    self._count += len(block)
    self._reservation = None
    if self._count >= self._rows:
      self.flush()

  def __getitem__(self, key):
    # Existing diagnostics must see all logical samples, including buffered tails.
    self.flush()
    return self._dataset[key]

  def flush(self):
    if self._reservation is not None:
      raise RuntimeError("cannot flush an incomplete append")
    if not self._count:
      return
    values = np.concatenate(self._pending, axis=0)
    start = self._size - self._count
    self._dataset.resize(self._size, axis=0)
    self._dataset[start : self._size] = values
    self._pending.clear()
    self._count = 0
    self.flush_count += 1


class CachedGroup:
  def __init__(self, group, owner):
    self._group = group
    self._owner = owner
    self._children = {}

  def __getattr__(self, name):
    return getattr(self._group, name)

  def __contains__(self, name):
    return name in self._children or name in self._group

  def __getitem__(self, name):
    if name not in self._children:
      self._children[name] = self._owner.wrap(self._group[name])
    return self._children[name]

  def create_group(self, name, *args, **kwargs):
    result = self._owner.wrap(self._group.create_group(name, *args, **kwargs))
    self._children[name] = result
    return result

  def create_dataset(self, name, *args, **kwargs):
    result = self._owner.wrap(self._group.create_dataset(name, *args, **kwargs))
    self._children[name] = result
    return result


class BufferedH5File(CachedGroup):
  """Cache handles and batch non-camera appends, defaulting to 64-row batches.

  ``rows`` bounds pending rows between completed appends. A single ragged-contact
  append may exceed this bound; it is flushed immediately. Abrupt SIGKILL or power
  loss can lose unflushed data, as with other in-process buffering; no durability
  guarantee is implied. Incomplete reservations fail closed, never publish zeros.
  """

  def __init__(self, file, rows=64):
    if type(rows) is not int or rows < 1:
      raise ValueError("buffer rows must be a positive integer")
    self._rows = rows
    self._objects = {}
    self._buffers = []
    super().__init__(file, self)

  def wrap(self, value):
    name = value.name
    if name in self._objects:
      return self._objects[name]
    if isinstance(value, h5py.Group):
      result = CachedGroup(value, self)
    elif (
      isinstance(value, h5py.Dataset)
      and value.ndim > 0
      and value.maxshape[0] is None
      and not name.startswith("/cameras/")
    ):
      result = BufferedDataset(value, self._rows)
      self._buffers.append(result)
    else:
      result = value
    self._objects[name] = result
    return result

  def flush(self):
    for dataset in self._buffers:
      dataset.flush()
    self._group.flush()

  def close(self):
    try:
      self.flush()
    finally:
      self._group.close()

  def statistics(self):
    return {
      "buffer_rows": self._rows,
      "buffered_streams": len(self._buffers),
      "batch_writes": sum(dataset.flush_count for dataset in self._buffers),
      "pending_rows": sum(dataset._count for dataset in self._buffers),
      "camera_writes": "immediate; RGB pixels and compression unchanged",
    }
