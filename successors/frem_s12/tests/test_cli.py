from pathlib import Path

import pytest

from frem_s12.cli import main


@pytest.mark.parametrize(
    ("command", "campaign_name", "message"),
    (
        ("select-search", "search", "selection output must be outside"),
        ("analyze", "fixed", "analysis output must be outside"),
    ),
)
def test_campaign_derived_output_cannot_mutate_campaign(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    campaign_name: str,
    message: str,
) -> None:
    campaign = tmp_path / campaign_name
    campaign.mkdir()
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                command,
                "--campaign",
                str(campaign),
                "--output",
                str(campaign / "derived.json"),
            ]
        )
    assert message in capsys.readouterr().err


def test_verification_output_cannot_mutate_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "verify-run-bundle",
                "--bundle",
                str(bundle),
                "--output",
                str(bundle / "verification.json"),
            ]
        )
    assert "verification output must be outside" in capsys.readouterr().err
