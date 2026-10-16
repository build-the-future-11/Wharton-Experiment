"""Engineering checks, including simulated OS counters; no market fits."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from successors.frem_s6_real import run
from successors.frem_s6_real.verify_data import verify


class RealIntakeTests(unittest.TestCase):
    def test_real_rows_reconcile(self):
        self.assertEqual(verify(Path('data/s6_real_etf_20261014'))['real_rows_reconciled'],1770)

    def test_frozen_identities(self):
        run.verify(Path('data/s6_real_etf_20261014'),Path('research/frem_s6_real_20261014/freeze2'))

    def test_low_memory_rejected(self):
        with patch.object(run,'snapshot',return_value={'available_mib':511,'pageouts':0}),patch.object(run.time,'sleep'):
            self.assertFalse(run.admission()['admitted'])

    def test_rising_swap_rejected(self):
        rows=[{'available_mib':1024,'pageouts':i} for i in range(3)]
        with patch.object(run,'snapshot',side_effect=rows),patch.object(run.time,'sleep'):
            self.assertFalse(run.admission()['admitted'])

    def test_stable_memory_admitted(self):
        with patch.object(run,'snapshot',return_value={'available_mib':512,'pageouts':0}),patch.object(run.time,'sleep'):
            self.assertTrue(run.admission()['admitted'])

    def test_linux_cgroup_headroom_limits_host_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            files={'/proc/meminfo':'MemAvailable: 8388608 kB\n',
                   '/proc/vmstat':'pswpout 4\n','/proc/self/cgroup':'0::/job\n',
                   '/sys/fs/cgroup/memory.max':'max','/sys/fs/cgroup/memory.current':'0',
                   '/sys/fs/cgroup/job/memory.max':str(1024*2**20),
                   '/sys/fs/cgroup/job/memory.current':str(800*2**20)}
            for name,value in files.items():
                target=base/name.lstrip('/')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(value)
            with patch.object(run.platform,'system',return_value='Linux'),patch.object(run,'Path',side_effect=lambda p:base/str(p).lstrip('/')):
                self.assertEqual(run.snapshot()['available_mib'],224)


if __name__=='__main__':
    unittest.main()
