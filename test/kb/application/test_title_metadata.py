"""Tests for deriving a release year from a JDIH document title."""
from __future__ import annotations

from datetime import datetime, timezone

from app.kb.application.title_metadata import parse_released_date, parse_released_year


class TestParseReleasedYear:
    def test_dashed_decree_number(self) -> None:
        title = "1313-UN40-KM.02.02-2026 - Peserta Program Outbound Student Mobility"
        assert parse_released_year(title) == 2026

    def test_nomor_tahun_form(self) -> None:
        title = "003 Tahun 2022 - Kelompok Kemampuan Ekonomi Orang Tua-Wali Calon Mahasiswa"
        assert parse_released_year(title) == 2022

    def test_academic_year_in_subject_is_ignored(self) -> None:
        # The subject's "Tahun Akademik 2025/2026" is not the issuing year;
        # only the number prefix is parsed.
        title = "2529-UN40-KM.02.02-2025 - Peserta Program Ke Universiti Malaysia Sabah Pada Semester Ganjil Tahun Akademik 2025/2026"
        assert parse_released_year(title) == 2025

    def test_last_year_in_the_prefix_wins(self) -> None:
        title = "41 Tahun 2021 - Perubahan Atas Peraturan Rektor Nomor 041 tahun 2020"
        assert parse_released_year(title) == 2021

    def test_no_year_returns_none(self) -> None:
        assert parse_released_year("Pedoman Umum Tanpa Nomor") is None

    def test_year_only_in_subject_returns_none(self) -> None:
        # Nothing parseable in the number block — better None than a wrong date.
        assert parse_released_year("Pedoman - Berlaku sejak 2019") is None

    def test_implausible_year_rejected(self) -> None:
        assert parse_released_year("9999-UN40-2999 - Sesuatu", max_year=2026) is None
        assert parse_released_year("1200-1200 - Sesuatu") is None

    def test_future_year_beyond_ceiling_rejected(self) -> None:
        assert parse_released_year("1-UN40-2030 - Sesuatu", max_year=2026) is None

    def test_empty_input(self) -> None:
        assert parse_released_year("") is None
        assert parse_released_year("   ") is None


class TestParseReleasedDate:
    def test_year_precision_is_january_first(self) -> None:
        assert parse_released_date("003 Tahun 2022 - Apa Saja") == datetime(
            2022, 1, 1, tzinfo=timezone.utc
        )

    def test_is_timezone_aware_utc(self) -> None:
        # released_date is a timestamptz. A naive midnight is read in the
        # session zone, so on UTC+7 it stores as 31 Dec of the previous year
        # and every document reads a year older than it is.
        parsed = parse_released_date("1313-UN40-KM.02.02-2026 - Apa Saja")
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0
        assert parsed.year == 2026

    def test_none_passes_through(self) -> None:
        assert parse_released_date("Tanpa Tahun") is None
