"""Namespace contract adverse fixtures; no numerical imports."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('guard', HERE/'namespace_guard.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def rows():
    base = dict(monotonic=1., membership='0::/', mount='fixture', maximum=30*2**30,
                high=30*2**30, current=2*2**30, available_bytes=28*2**30,
                pswpout=0, events={'high':0,'max':0,'oom':0,'oom_kill':0})
    return [dict(copy.deepcopy(base), monotonic=float(i)) for i in (1,2,3)]


class NamespaceTests(unittest.TestCase):
    def test_admission_never_claims_full_root_or_science(self):
        r = g.assess(rows())
        self.assertEqual(r['status'],'SMOKE_ONLY_ADMITTED')
        self.assertFalse(r['production_admission'])
        self.assertEqual(r['hidden_ancestor_constraints'],'UNKNOWN')

    def test_insufficient_headroom_rejected(self):
        r=rows();r[1]['available_bytes']=1536*2**20-1
        with self.assertRaises(RuntimeError): g.assess(r)

    def test_changed_identity_pressure_and_spacing_rejected(self):
        for key,value in [('membership','other'),('mount','other'),('maximum',1),('high',1),
                          ('pswpout',1),('events',{'oom':1}),('monotonic',1.5)]:
            r=rows();r[1][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError): g.assess(r)

    def test_missing_samples_rejected(self):
        with self.assertRaises(RuntimeError):g.assess(rows()[:2])

    def test_unlimited_negative_nonfinite_or_duplicate_counter_rejected(self):
        for v in ['max','-1','nan','1.0']:
            with self.assertRaises(RuntimeError):g.integer(v)
        with self.assertRaises(RuntimeError):g.pairs('oom 0\noom 1')

    def test_production_commands_rejected(self):
        for name in ['advance','run','pilot','qualification','preflight']:
            with self.assertRaises(RuntimeError):g.validate_command([name])

    def test_only_sealed_twelve_origin_manifest(self):
        import zipfile
        archive=HERE.parents[1]/'research/frem_s10_kaggle_20261016/source_bundle.zip'
        with zipfile.ZipFile(archive) as z,tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json';p.write_bytes(z.read('configs/generated_smoke.json'))
            g.validate_command(['smoke','--manifest',str(p)])
            value=json.loads(p.read_text());value['steps']=13;p.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError):g.validate_command(['smoke','--manifest',str(p)])

    def test_limits_are_never_raised(self):
        with patch.object(g.resource,'getrlimit',return_value=(100,200)),patch.object(g.resource,'setrlimit') as set_limit:
            g.hard_limits()
        self.assertEqual(set_limit.call_args_list[0].args[1],(100,100))
        self.assertEqual(set_limit.call_args_list[1].args[1],(75,75))

    def test_actual_namespace_layout_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);proc=p/'proc';cg=p/'cg';(proc/'self').mkdir(parents=True);cg.mkdir()
            (proc/'self/cgroup').write_text('0::/\n')
            (proc/'self/mountinfo').write_text(f'1 2 0:1 / {cg} ro - cgroup2 cgroup rw\n')
            (proc/'meminfo').write_text('MemAvailable: 31446316 kB\n')
            (proc/'vmstat').write_text('pswpout 0\n')
            for name,value in {'cgroup.controllers':'memory cpu','memory.max':'32212254720','memory.current':'1863487488','memory.high':'max','memory.events':'high 0\nmax 0\noom 0\noom_kill 0'}.items():
                (cg/name).write_text(value)
            r=g.sample(proc,cg)
            self.assertEqual(r['available_bytes'],32212254720-1863487488)
            (cg/'memory.max').write_text('max')
            with self.assertRaises(RuntimeError):g.sample(proc,cg)


if __name__=='__main__':unittest.main()
