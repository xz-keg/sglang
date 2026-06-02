import shutil
import tempfile
import unittest
import uuid

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.mem_cache.memory_pool_host import (
    CompactHostSlotAllocator,
    SharedMemoryHostTensorAllocator,
)
from sglang.test.test_utils import CustomTestCase


class TestSharedMemoryHostTensorAllocator(CustomTestCase):
    def test_same_prefix_maps_same_storage(self):
        directory = tempfile.mkdtemp()
        prefix = f"allocator_unit_{uuid.uuid4().hex}"
        try:
            first = SharedMemoryHostTensorAllocator(
                group=None,
                name_prefix=prefix,
                directory=directory,
                unlink_after_attach=False,
            )
            second = SharedMemoryHostTensorAllocator(
                group=None,
                name_prefix=prefix,
                directory=directory,
                unlink_after_attach=False,
            )

            a = first.allocate((8,), torch.int32, "cpu")
            b = second.allocate((8,), torch.int32, "cpu")

            self.assertTrue(first.is_writer)
            self.assertFalse(second.is_writer)
            a.copy_(torch.arange(8, dtype=torch.int32))
            self.assertTrue(torch.equal(b, torch.arange(8, dtype=torch.int32)))
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class TestCompactHostSlotAllocator(CustomTestCase):
    def test_matches_queue_order_without_full_free_list(self):
        allocator = CompactHostSlotAllocator(16)

        first = allocator.alloc(6)
        second = allocator.alloc(4)
        self.assertTrue(torch.equal(first, torch.arange(6, dtype=torch.int64)))
        self.assertTrue(torch.equal(second, torch.arange(6, 10, dtype=torch.int64)))

        allocator.free(first[2:6])
        third = allocator.alloc(8)
        self.assertTrue(
            torch.equal(third, torch.tensor([10, 11, 12, 13, 14, 15, 2, 3]))
        )
        self.assertEqual(allocator.available_size(), 2)


if __name__ == "__main__":
    unittest.main()
