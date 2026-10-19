from frem_s11.acquisition import declared_sources


def test_declared_archive_formats_match_official_index() -> None:
    sources = declared_sources()
    assert sources[200808].endswith("6_Portfolios_2x3_TXT.zip")
    assert sources[201608].endswith("6_Portfolios_2x3_CSV.zip")
    assert sources[202408].endswith("6_Portfolios_2x3_CSV.zip")
