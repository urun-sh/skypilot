"""Regression tests for the Spheron catalog fetcher's CSV writer.

The prod catalog refresh for Spheron has been silently failing with
FileNotFoundError: ``catalog.common.get_catalog_path`` is a pure path
getter (no side effects), and ``write_csv`` opened the output without
creating its parent directory. These tests pin the fix: the writer must
create the output's parent directory itself, matching fetch_vast.py.
"""
import csv
import os
import tempfile
import unittest

from sky.catalog.data_fetchers import fetch_spheron


class TestWriteCsv(unittest.TestCase):
    """write_csv must survive an output path whose parent does not exist."""

    def test_creates_missing_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'spheron', 'vms.csv')
            # Only tmp exists; 'spheron/' does not.
            self.assertFalse(os.path.exists(os.path.dirname(out)))
            written = fetch_spheron.write_csv([], out)
            self.assertEqual(written, 0)
            self.assertTrue(os.path.exists(out))

    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'nested', 'spheron', 'vms.csv')
            row = ['spheron-gpu-x1'
                  ] + [''] * (len(fetch_spheron.CSV_COLUMNS) - 1)
            written = fetch_spheron.write_csv([row], out)
            self.assertEqual(written, 1)
            with open(out, encoding='utf-8', newline='') as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], fetch_spheron.CSV_COLUMNS)
            self.assertEqual(rows[1], row)


if __name__ == '__main__':
    unittest.main()
