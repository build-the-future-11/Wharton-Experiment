from frem_s11.admission import (
    MemorySample,
    assess,
    execution_pressure_stable,
    parse_vm_stat,
)

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               40000.
Pages speculative:                        1000.
Pageouts:                                  222.
"""


def test_admission_uses_three_samples_and_stable_pageouts() -> None:
    rows = tuple(parse_vm_stat(VM_STAT, timestamp_utc=f"t{index}") for index in range(3))
    receipt = assess(rows)
    assert receipt["status"] == "PASS"
    assert execution_pressure_stable(receipt, rows[-1])
    changed = MemorySample("later", 16384, 40000, 1000, 223)
    assert not execution_pressure_stable(receipt, changed)


def test_admission_blocks_below_floor() -> None:
    sample = MemorySample("t", 16384, 1, 1, 0)
    receipt = assess((sample, sample, sample))
    assert receipt["status"] == "BLOCKED"
    assert "AVAILABLE_MEMORY_BELOW_512_MIB" in receipt["reasons"]
