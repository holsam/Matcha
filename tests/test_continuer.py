"""
Tests for the continue command and related configuration management.

Tests cover config saving/loading, staleness detection, FAISS invalidation,
and the main continue command flow.
"""

import click, json, os, pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

from matcha.config import save_run_config, load_run_config
from matcha.continuer import (
    get_available_configs,
    check_index_stale,
    invalidate_faiss_index,
    run_continue,
)


# ============================================================================
# TestConfigSaving
# ============================================================================


class TestConfigSaving:
    """Tests for saving run configurations."""

    def test_index_saves_config(self, indexed_dir):
        """After indexing, .matcha/index.json should exist with correct structure."""
        matcha_dir = os.path.join(indexed_dir["dir"], ".matcha")
        config_path = Path(matcha_dir, "index.json")

        assert os.path.exists(config_path), "Config file should exist after indexing"

        with open(config_path, "r") as f:
            config = json.load(f)

        assert config.get("command") == "index"
        assert "last_run" in config
        assert "args" in config
        assert all(
            k in config["args"]
            for k in ["fps", "workers", "no_audio", "hwaccel"]
        )

    def test_match_saves_config(self, matched_dir):
        """After matching, .matcha/match.json should exist with correct structure."""
        matcha_dir = os.path.join(matched_dir["dir"], ".matcha")
        config_path = os.path.join(matcha_dir, "match.json")

        assert os.path.exists(config_path), "Config file should exist after matching"

        with open(config_path, "r") as f:
            config = json.load(f)

        assert config.get("command") == "match"
        assert "last_run" in config
        assert "args" in config
        assert all(
            k in config["args"]
            for k in [
                "filter_length",
                "window",
                "frame_step",
                "threshold",
                "min_confidence",
                "workers",
                "nprobe",
            ]
        )

    def test_config_args_match_call(self, tmp_path):
        """Saved args should match the parameters passed to save_run_config."""
        matcha_dir = os.path.join(tmp_path, ".matcha")

        test_args = {
            "fps": 2.0,
            "workers": 2,
            "no_audio": True,
            "hwaccel": False,
        }

        save_run_config(matcha_dir, "index", test_args)

        config = load_run_config(matcha_dir, "index")
        assert config is not None
        assert config["args"] == test_args

    def test_config_saved_atomically(self, tmp_path):
        """Config should be written atomically (temp file then replace)."""
        matcha_dir = os.path.join(tmp_path, ".matcha")
        config_path = os.path.join(matcha_dir, "index.json")

        # First write
        save_run_config(matcha_dir, "index", {"fps": 1.0, "workers": 4})
        assert os.path.exists(config_path)

        # Second write should replace cleanly
        save_run_config(matcha_dir, "index", {"fps": 2.0, "workers": 8})
        with open(config_path) as f:
            config = json.load(f)
        assert config["args"]["fps"] == 2.0


# ============================================================================
# TestLoadConfig
# ============================================================================


class TestLoadConfig:
    """Tests for loading run configurations."""

    def test_load_returns_none_when_missing(self, tmp_path):
        """load_run_config should return None if file doesn't exist."""
        matcha_dir = os.path.join(tmp_path, ".matcha")
        result = load_run_config(matcha_dir, "index")
        assert result is None

    def test_load_returns_none_on_malformed_json(self, tmp_path):
        """load_run_config should return None for invalid JSON without raising."""
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)
        config_path = os.path.join(matcha_dir, "index.json")

        with open(config_path, "w") as f:
            f.write("{ invalid json }")

        result = load_run_config(matcha_dir, "index")
        assert result is None

    def test_load_returns_dict_when_valid(self, tmp_path):
        """load_run_config should return parsed dict for valid config."""
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)
        config_path = os.path.join(matcha_dir, "index.json")

        test_config = {
            "command": "index",
            "last_run": "2025-11-01T14:32:00Z",
            "args": {"fps": 1.0, "workers": 4},
        }

        with open(config_path, "w") as f:
            json.dump(test_config, f)

        result = load_run_config(matcha_dir, "index")
        assert result == test_config


# ============================================================================
# TestGetAvailableConfigs
# ============================================================================


class TestGetAvailableConfigs:
    """Tests for discovering available configurations."""

    def test_no_configs_returns_empty(self, tmp_path):
        """Should return empty dict if no config files exist."""
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)

        result = get_available_configs(matcha_dir)
        assert result == {}

    def test_returns_only_valid_configs(self, tmp_path):
        """Should skip malformed configs and return only valid ones."""
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)

        # Valid index config
        with open(os.path.join(matcha_dir, "index.json"), "w") as f:
            json.dump(
                {
                    "command": "index",
                    "last_run": "2025-11-01T14:32:00Z",
                    "args": {},
                },
                f,
            )

        # Malformed match config
        with open(os.path.join(matcha_dir, "match.json"), "w") as f:
            f.write("{ broken json }")

        result = get_available_configs(matcha_dir)
        assert "index" in result
        assert "match" not in result

    def test_returns_both_when_present(self, tmp_path):
        """Should return both configs if both exist and are valid."""
        matcha_dir = os.path.join(tmp_path, ".matcha")

        save_run_config(matcha_dir, "index", {"fps": 1.0})
        save_run_config(matcha_dir, "match", {"window": 10.0})

        result = get_available_configs(matcha_dir)
        assert "index" in result
        assert "match" in result


# ============================================================================
# TestStaleIndexCheck
# ============================================================================


class TestStaleIndexCheck:
    """Tests for detecting if the index has been updated since last match."""

    def test_not_stale_when_index_predates_match(self, tmp_path):
        """Index is not stale if all fingerprints predate the match last_run."""
        matcha_dir, db_path = self._setup_test_db(tmp_path)

        # Insert a video fingerprinted 1 hour ago
        past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        conn = self._get_db_conn(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO videos (path, duration, fingerprinted_at) VALUES (?, ?, ?)",
            ("test.mp4", 100.0, past_time),
        )
        conn.commit()
        conn.close()

        # Match config with a recent last_run (just now)
        match_config = {
            "last_run": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

        result = check_index_stale(matcha_dir, match_config)
        assert result is False

    def test_stale_when_video_fingerprinted_after_match(self, tmp_path):
        """Index is stale if any fingerprint is newer than match last_run."""
        matcha_dir, db_path = self._setup_test_db(tmp_path)

        # Match config with a last_run 1 hour ago
        past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        match_config = {"last_run": past_time.replace("+00:00", "Z")}

        # Insert a video fingerprinted just now
        recent_time = datetime.now(timezone.utc).isoformat()
        conn = self._get_db_conn(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO videos (path, duration, fingerprinted_at) VALUES (?, ?, ?)",
            ("test.mp4", 100.0, recent_time),
        )
        conn.commit()
        conn.close()

        result = check_index_stale(matcha_dir, match_config)
        assert result is True

    def test_stale_when_db_empty(self, tmp_path):
        """Index is not stale if DB is empty (no videos to warn about)."""
        matcha_dir, db_path = self._setup_test_db(tmp_path)

        match_config = {
            "last_run": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

        result = check_index_stale(matcha_dir, match_config)
        assert result is False

    # Helper methods
    def _setup_test_db(self, tmp_path):
        """Create a minimal test database with required tables."""
        matcha_dir = os.path.join(tmp_path, ".matcha")
        db_path = os.path.join(tmp_path, "index.db")
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)

        conn = self._get_db_conn(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                duration REAL,
                fingerprinted_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

        return matcha_dir, db_path

    @staticmethod
    def _get_db_conn(db_path):
        """Get a database connection (assumes sqlite3)."""
        import sqlite3

        return sqlite3.connect(db_path)


# ============================================================================
# TestInvalidateFaissIndex
# ============================================================================


class TestInvalidateFaissIndex:
    """Tests for invalidating the FAISS index."""

    def test_deletes_faiss_files(self, tmp_path):
        """Should delete frame_index.faiss and frame_index_map.npy."""
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)

        faiss_file = os.path.join(matcha_dir, "frame_index.faiss")
        faiss_map = os.path.join(matcha_dir, "frame_index_map.npy")

        # Create dummy files
        Path(faiss_file).touch()
        Path(faiss_map).touch()

        invalidate_faiss_index(matcha_dir)

        assert not os.path.exists(faiss_file)
        assert not os.path.exists(faiss_map)

    def test_clears_faiss_meta_table(self, tmp_path):
        """Should clear the faiss_index_meta table."""
        matcha_dir = os.path.join(tmp_path, ".matcha")
        db_path = os.path.join(tmp_path, "index.db")
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)

        # Create DB with faiss_index_meta table
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faiss_index_meta (
                id INTEGER PRIMARY KEY,
                built_at TEXT
            )
            """
        )
        cursor.execute("INSERT INTO faiss_index_meta (built_at) VALUES (?)", ("2025-01-01",))
        conn.commit()
        conn.close()

        # Invalidate
        invalidate_faiss_index(matcha_dir)

        # Check table is empty
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM faiss_index_meta")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0

    def test_no_error_when_files_missing(self, tmp_path):
        """Should not raise if FAISS files don't exist."""
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)

        # Should not raise
        invalidate_faiss_index(matcha_dir)


# ============================================================================
# TestRunContinue
# ============================================================================


class TestRunContinue:
    """Integration tests for the continue command."""

    def test_exits_when_no_db(self, tmp_path):
        """Should exit cleanly if no database exists."""
        with pytest.raises((click.exceptions.Exit, SystemExit)) as exc_info:
            run_continue(str(tmp_path))
        assert exc_info.value.exit_code == 1

    def test_exits_when_no_configs(self, tmp_path):
        """Should exit cleanly if no config files exist."""
        db_path = os.path.join(tmp_path, "index.db")
        Path(db_path).touch()
        with pytest.raises((click.exceptions.Exit,SystemExit)) as exc_info:
            run_continue(str(tmp_path))
        assert exc_info.value.exit_code == 1

    def test_exits_when_specified_config_missing(self, tmp_path):
        """Should exit if requested command has no saved config."""
        db_path = os.path.join(tmp_path, "index.db")
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(db_path).touch()
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)
        # Save only index config
        save_run_config(matcha_dir, "index", {"fps": 1.0})
        # Try to continue match (which doesn't exist)
        with pytest.raises((click.exceptions.Exit, SystemExit)) as exc_info:
            run_continue(str(tmp_path), "match")
        assert exc_info.value.exit_code == 1

    def test_continues_index_with_saved_args(self, tmp_path):
        """Should re-run index with the saved args."""
        # Setup a minimal indexed directory
        db_path = os.path.join(tmp_path, "index.db")
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(db_path).touch()
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)

        # Save a config with specific args
        save_run_config(
            matcha_dir,
            "index",
            {"fps": 2.0, "workers": 2, "no_audio": True, "hwaccel": False},
        )

        # Mock run_index to verify it gets called with the right args
        with patch("matcha.continuer.run_index") as mock_run_index:
            run_continue(str(tmp_path), "index")

            mock_run_index.assert_called_once()
            # Check that fps=2.0 was passed
            call_args = mock_run_index.call_args
            assert call_args[1].get("fps") == 2.0
            assert call_args[1].get("workers") == 2

    def test_continues_match_with_saved_args(self, tmp_path):
        """Should re-run match with the saved args."""
        db_path = os.path.join(tmp_path, "index.db")
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(db_path).touch()
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)

        # Save a config with specific args
        save_run_config(
            matcha_dir,
            "match",
            {
                "filter_length": True,
                "window": 20.0,
                "frame_step": 5,
                "threshold": 15,
                "min_confidence": 0.7,
                "workers": 2,
                "nprobe": 64,
            },
        )

        # Mock run_match to verify it gets called with the right args
        with patch("matcha.continuer.run_match") as mock_run_match:
            run_continue(str(tmp_path), "match")

            mock_run_match.assert_called_once()
            call_args = mock_run_match.call_args
            assert call_args[1].get("window") == 20.0
            assert call_args[1].get("frame_step") == 5

    def test_stale_index_triggers_faiss_rebuild(self, tmp_path):
        """Should invalidate FAISS index if videos added since last match."""
        db_path = os.path.join(tmp_path, "index.db")
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(matcha_dir).mkdir(parents=True, exist_ok=True)

        # Create DB and add a video
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY,
                path TEXT,
                duration REAL,
                fingerprinted_at TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faiss_index_meta (
                id INTEGER PRIMARY KEY,
                built_at TEXT
            )
            """
        )
        # Video fingerprinted recently (after the match config's last_run)
        recent_time = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "INSERT INTO videos (path, duration, fingerprinted_at) VALUES (?, ?, ?)",
            ("test.mp4", 100.0, recent_time),
        )
        cursor.execute("INSERT INTO faiss_index_meta (built_at) VALUES (?)", ("2025-01-01",))
        conn.commit()
        conn.close()

        # Match config with a last_run from 1 hour ago
        past_time = (
            (datetime.now(timezone.utc) - timedelta(hours=1))
            .isoformat()
            .replace("+00:00", "Z")
        )
        save_run_config(
            matcha_dir,
            "match",
            {
                "filter_length": False,
                "window": 10.0,
                "frame_step": 3,
                "threshold": 10,
                "min_confidence": 0.8,
                "workers": 4,
                "nprobe": 32,
            },
        )

        # Manually update last_run to be in the past
        config_path = os.path.join(matcha_dir, "match.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        config["last_run"] = past_time
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Create dummy FAISS files
        faiss_file = os.path.join(matcha_dir, "frame_index.faiss")
        faiss_map = os.path.join(matcha_dir, "frame_index_map.npy")
        Path(faiss_file).touch()
        Path(faiss_map).touch()

        # Mock run_match
        with patch("matcha.continuer.run_match"):
            run_continue(str(tmp_path), "match")

        # FAISS files should be deleted
        assert not os.path.exists(faiss_file)
        assert not os.path.exists(faiss_map)

    def test_prompts_when_both_configs_present(self, tmp_path):
        """Should prompt user when both index and match configs exist."""
        db_path = os.path.join(tmp_path, "index.db")
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(db_path).touch()

        save_run_config(matcha_dir, "index", {"fps": 1.0})
        save_run_config(matcha_dir, "match", {"window": 10.0})

        # Mock typer.prompt to select "index" (choice 1)
        with patch("matcha.continuer.typer.prompt", return_value="1"):
            with patch("matcha.continuer.run_index"):
                run_continue(str(tmp_path), None)

    def test_no_prompt_when_one_config_present(self, tmp_path):
        """Should not prompt if only one config exists."""
        db_path = os.path.join(tmp_path, "index.db")
        matcha_dir = os.path.join(tmp_path, ".matcha")
        Path(db_path).touch()

        save_run_config(matcha_dir, "index", {"fps": 1.0})

        # Mock input to ensure it's not called
        with patch("matcha.continuer.typer.prompt", side_effect=AssertionError("Should not prompt")):
            with patch("matcha.continuer.run_index"):
                run_continue(str(tmp_path), None)