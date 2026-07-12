"""Durable candidate state and mutation recovery."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.repository_index.overlay import CandidateOverlay
from kazusa_ai_chatbot.coding_agent.safety import (
    confined_managed_repo_path,
    copy_managed_source_tree,
    normalize_safe_repo_relative_path,
)


_TERMINAL_OPERATION_STATES = {"committed", "rolled_back"}


@dataclass
class CandidateState:
    """Own one candidate root and its resumable mutation journal."""

    root: Path
    revision: int
    journal: list[dict[str, object]]

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        source_root: Path | None = None,
    ) -> "CandidateState":
        """Create one candidate root from a resolved source or empty baseline."""

        if root.is_symlink():
            raise ValueError("candidate root is unsafe")
        candidate_source = root / "source"
        candidate_source.mkdir(parents=True, exist_ok=True)
        if source_root is not None:
            copy_managed_source_tree(
                source_root,
                candidate_source,
                dirs_exist_ok=True,
            )
        state = cls(root=root, revision=0, journal=[])
        state._save()
        return state

    @classmethod
    def load(cls, root: Path) -> "CandidateState":
        """Load persisted candidate state before a resumed action."""

        if root.is_symlink() or (root / "source").is_symlink():
            raise ValueError("persisted candidate root is unsafe")
        payload = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("revision"), int)
            or isinstance(payload.get("revision"), bool)
            or not isinstance(payload.get("journal"), list)
            or not all(
                isinstance(row, dict)
                for row in payload.get("journal", [])
            )
        ):
            raise ValueError("persisted candidate state is invalid")
        state = cls(
            root=root,
            revision=payload["revision"],
            journal=payload["journal"],
        )
        return state

    def recover(self) -> list[dict[str, str]]:
        """Reconcile interrupted candidate and overlay phases idempotently."""

        outcomes: list[dict[str, str]] = []
        for operation in self.journal:
            state = operation["state"]
            if state in _TERMINAL_OPERATION_STATES:
                self._cleanup_backups(operation)
                continue
            self._validate_recovery_metadata(operation)
            if state == "prepared":
                self._roll_back(operation)
                outcomes.append({
                    "operation_id": str(operation["operation_id"]),
                    "state": "rolled_back",
                })
                continue
            if state == "candidate_written":
                if not self._candidate_matches_result(operation):
                    self._roll_back(operation)
                    outcomes.append({
                        "operation_id": str(operation["operation_id"]),
                        "state": "rolled_back",
                    })
                    raise ValueError(
                        "candidate recovery failed: candidate result identity "
                        "mismatch",
                    )
                self._write_result_overlay(operation)
                operation["state"] = "overlay_written"
                self._save()
                state = "overlay_written"
            if state == "overlay_written":
                if (
                    not self._candidate_matches_result(operation)
                    or not self._overlay_matches_result(operation)
                ):
                    self._roll_back(operation)
                    outcomes.append({
                        "operation_id": str(operation["operation_id"]),
                        "state": "rolled_back",
                    })
                    raise ValueError(
                        "candidate recovery failed: result identity mismatch",
                    )
                self._commit_operation(operation)
                outcomes.append({
                    "operation_id": str(operation["operation_id"]),
                    "state": "committed",
                })
                continue
            raise ValueError("candidate recovery journal state is invalid")
        return outcomes

    def require_recovered_before_next_action(self) -> None:
        """Reject dispatch while any durable candidate mutation is unresolved."""

        if any(
            row.get("state") not in _TERMINAL_OPERATION_STATES
            for row in self.journal
        ):
            raise ValueError("candidate recovery is required before the next action")

    def apply_journaled_mutation(
        self,
        *,
        operation_id: str,
        kind: str,
        repo_path: str,
        replacement: str | None,
        expected_revision: int,
        expected_source_sha256: str | None,
        target_path: str | None = None,
    ) -> dict[str, object]:
        """Apply candidate and overlay changes through one four-state journal."""

        self.require_recovered_before_next_action()
        if (
            not operation_id
            or not operation_id.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError("candidate operation identity is invalid")
        safe_repo_path = self._validated_repo_path(repo_path)
        safe_target_path = (
            self._validated_repo_path(target_path)
            if target_path is not None
            else None
        )
        allowed_kinds = {
            "create_file",
            "replace_anchor",
            "insert_before",
            "insert_after",
            "replace_file_small",
            "delete_file",
            "rename_file",
        }
        if kind not in allowed_kinds:
            raise ValueError("candidate mutation kind is unsupported")
        if kind != "rename_file" and safe_target_path is not None:
            raise ValueError("candidate mutation target path is unsupported")
        if kind == "create_file" and expected_source_sha256 is not None:
            raise ValueError("candidate create source identity is invalid")
        if kind in {"delete_file", "rename_file"} and replacement is not None:
            raise ValueError("candidate mutation replacement is invalid")
        result_sha256 = (
            expected_source_sha256
            if kind == "rename_file"
            else (
                hashlib.sha256(replacement.encode("utf-8")).hexdigest()
                if replacement is not None
                else None
            )
        )
        existing = next(
            (row for row in self.journal if row.get("operation_id") == operation_id),
            None,
        )
        if existing is not None:
            if existing.get("state") != "committed":
                raise ValueError("candidate recovery is required before replay")
            replay_identity = {
                "kind": kind,
                "repo_path": safe_repo_path,
                "target_path": safe_target_path,
                "expected_candidate_revision": expected_revision,
                "resulting_candidate_revision": expected_revision + 1,
                "expected_source_sha256": expected_source_sha256,
                "result_sha256": result_sha256,
            }
            if any(
                existing.get(key) != value
                for key, value in replay_identity.items()
            ):
                raise ValueError("candidate operation replay identity mismatch")
            return existing
        if expected_revision != self.revision:
            raise ValueError("candidate revision is stale")
        path = self._candidate_path(safe_repo_path)
        if kind == "rename_file":
            if safe_target_path is None:
                raise ValueError("candidate rename target is required")
            if safe_target_path.casefold() == safe_repo_path.casefold():
                raise ValueError("candidate rename target collides with source")
            target = self._candidate_path(safe_target_path)
            if target.exists() or target.is_symlink():
                raise ValueError("candidate rename target already exists")
        before = self._safe_file_bytes(path)
        if kind == "create_file" and before is not None:
            raise ValueError("candidate create target already exists")
        if kind != "create_file" and before is None:
            raise ValueError("candidate mutation source is missing")
        if kind != "create_file" and expected_source_sha256 is None:
            raise ValueError("candidate mutation source identity is missing")
        if kind not in {"delete_file", "rename_file"} and replacement is None:
            raise ValueError("candidate mutation replacement is missing")
        if expected_source_sha256 is not None:
            current_digest = None
            if before is not None:
                try:
                    current_text = before.decode("utf-8").replace(
                        "\r\n",
                        "\n",
                    ).replace("\r", "\n")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"candidate source is not safe text: {exc}",
                    ) from exc
                current_digest = hashlib.sha256(
                    current_text.encode("utf-8"),
                ).hexdigest()
            if current_digest != expected_source_sha256:
                raise ValueError("candidate source hash is stale")
        affected_paths = [safe_repo_path]
        if safe_target_path is not None:
            affected_paths.append(safe_target_path)
        overlay = CandidateOverlay(self.root / "overlay.sqlite")
        try:
            prior_overlay = overlay.describe_paths(affected_paths)
        finally:
            overlay.close()
        before_paths = self._write_backups(
            operation_id=operation_id,
            repo_paths=affected_paths,
        )
        resulting_revision = self.revision + 1
        operation = {
            "operation_id": operation_id,
            "kind": kind,
            "repo_path": safe_repo_path,
            "target_path": safe_target_path,
            "expected_candidate_revision": expected_revision,
            "resulting_candidate_revision": resulting_revision,
            "expected_source_sha256": expected_source_sha256,
            "result_sha256": result_sha256,
            "before_paths": before_paths,
            "prior_overlay": prior_overlay,
            "state": "prepared",
        }
        self.journal.append(operation)
        self._save()
        try:
            if kind == "delete_file":
                path.unlink()
            elif kind == "rename_file":
                target = self._candidate_path(str(safe_target_path))
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)
            else:
                self._write_candidate_file(
                    path,
                    (replacement or "").encode("utf-8"),
                )
            operation["state"] = "candidate_written"
            self._save()
            overlay = CandidateOverlay(self.root / "overlay.sqlite")
            try:
                if kind == "delete_file":
                    overlay.delete(
                        repo_path=safe_repo_path,
                        revision=resulting_revision,
                    )
                elif kind == "rename_file":
                    overlay.rename(
                        source_path=safe_repo_path,
                        target_path=str(safe_target_path),
                        content=self._safe_text(str(safe_target_path)),
                        revision=resulting_revision,
                    )
                else:
                    overlay.upsert(
                        repo_path=safe_repo_path,
                        content=self._safe_text(safe_repo_path),
                        revision=resulting_revision,
                    )
            finally:
                overlay.close()
            operation["state"] = "overlay_written"
            self._save()
            self._commit_operation(operation)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            if operation.get("state") not in _TERMINAL_OPERATION_STATES:
                self._roll_back(operation)
            raise ValueError(f"candidate mutation rolled back: {exc}") from exc
        return operation

    def _validate_recovery_metadata(
        self,
        operation: dict[str, object],
    ) -> None:
        """Validate every recovery identity before changing candidate state."""

        before_paths = operation.get("before_paths")
        prior_overlay = operation.get("prior_overlay")
        if not isinstance(before_paths, list) or not isinstance(prior_overlay, list):
            raise ValueError("candidate recovery backup identity is missing")
        if len(before_paths) not in {1, 2} or len(before_paths) != len(prior_overlay):
            raise ValueError("candidate recovery backup identity count is invalid")
        seen_paths: set[str] = set()
        backup_content_by_path: dict[str, bytes] = {}
        for before_path in before_paths:
            if not isinstance(before_path, dict):
                raise ValueError("candidate recovery backup identity is invalid")
            repo_path = before_path.get("repo_path")
            safe_path = self._validated_repo_path(repo_path)
            if safe_path in seen_paths:
                raise ValueError("candidate recovery backup path is duplicated")
            seen_paths.add(safe_path)
            existed = before_path.get("existed")
            expected_sha256 = before_path.get("sha256")
            backup_relative_path = before_path.get("backup_relative_path")
            if not isinstance(existed, bool):
                raise ValueError("candidate recovery backup identity is invalid")
            if not existed:
                if expected_sha256 is not None or backup_relative_path is not None:
                    raise ValueError(
                        "candidate recovery absent-path identity is invalid",
                    )
                candidate_path = self._candidate_path(safe_path)
                self._validate_current_recovery_path(candidate_path)
                continue
            if not isinstance(expected_sha256, str) or not isinstance(
                backup_relative_path,
                str,
            ):
                raise ValueError("candidate recovery backup identity is missing")
            backup_path = self._validated_backup_path(backup_relative_path)
            try:
                backup_content = backup_path.read_bytes()
            except OSError as exc:
                raise ValueError(
                    f"candidate recovery backup identity is missing: {exc}",
                ) from exc
            actual_sha256 = hashlib.sha256(backup_content).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError("candidate recovery backup identity mismatch")
            backup_content_by_path[safe_path] = backup_content
            candidate_path = self._candidate_path(safe_path)
            self._validate_current_recovery_path(candidate_path)
        prior_paths = []
        for index, overlay_record in enumerate(prior_overlay):
            if not isinstance(overlay_record, dict):
                raise ValueError("candidate recovery prior overlay is invalid")
            prior_path = self._validated_repo_path(
                overlay_record.get("repo_path"),
            )
            prior_paths.append(prior_path)
            overlay_state = overlay_record.get("state")
            content_sha256 = overlay_record.get("content_sha256")
            revision = overlay_record.get("revision")
            if overlay_state not in {
                "absent",
                "content",
                "tombstone",
            }:
                raise ValueError("candidate recovery prior overlay is invalid")
            if overlay_state == "absent":
                if content_sha256 is not None or revision is not None:
                    raise ValueError("candidate recovery prior overlay is invalid")
                continue
            if not isinstance(revision, int):
                raise ValueError("candidate recovery prior overlay is invalid")
            if overlay_state == "tombstone":
                if content_sha256 is not None:
                    raise ValueError("candidate recovery prior overlay is invalid")
                continue
            if not isinstance(content_sha256, str):
                raise ValueError("candidate recovery prior overlay is invalid")
            before_path = before_paths[index]
            if not before_path["existed"]:
                raise ValueError("candidate recovery prior content is unavailable")
            backup_content = backup_content_by_path[prior_path]
            try:
                normalized_content = backup_content.decode("utf-8").replace(
                    "\r\n",
                    "\n",
                ).replace("\r", "\n")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"candidate recovery prior content is invalid: {exc}",
                ) from exc
            actual_content_sha256 = hashlib.sha256(
                normalized_content.encode("utf-8"),
            ).hexdigest()
            if actual_content_sha256 != content_sha256:
                raise ValueError("candidate recovery prior overlay identity mismatch")
        if prior_paths != [row["repo_path"] for row in before_paths]:
            raise ValueError("candidate recovery prior overlay paths mismatch")

    def _write_backups(
        self,
        *,
        operation_id: str,
        repo_paths: list[str],
    ) -> list[dict[str, object]]:
        """Write and verify bounded recovery backups before journaling prepare."""

        recovery_directory = self._recovery_operation_path(operation_id)
        records: list[dict[str, object]] = []
        try:
            for index, repo_path in enumerate(repo_paths):
                candidate_path = self._candidate_path(repo_path)
                content = self._safe_file_bytes(candidate_path)
                if content is None:
                    records.append({
                        "repo_path": repo_path,
                        "existed": False,
                        "sha256": None,
                        "backup_relative_path": None,
                    })
                    continue
                backup_relative_path = (
                    Path("recovery") / operation_id / f"{index}.backup"
                )
                backup_path = confined_managed_repo_path(
                    self.root / "recovery",
                    (Path(operation_id) / f"{index}.backup").as_posix(),
                )
                self._write_candidate_file(backup_path, content)
                expected_sha256 = hashlib.sha256(content).hexdigest()
                actual_sha256 = hashlib.sha256(backup_path.read_bytes()).hexdigest()
                if actual_sha256 != expected_sha256:
                    raise ValueError("candidate recovery backup write mismatch")
                records.append({
                    "repo_path": repo_path,
                    "existed": True,
                    "sha256": expected_sha256,
                    "backup_relative_path": backup_relative_path.as_posix(),
                })
        except (OSError, ValueError):
            shutil.rmtree(recovery_directory, ignore_errors=True)
            raise
        return records

    def _candidate_matches_result(self, operation: dict[str, object]) -> bool:
        """Return whether all affected candidate paths match the result identity."""

        kind = operation["kind"]
        repo_path = operation["repo_path"]
        target_path = operation["target_path"]
        result_sha256 = operation["result_sha256"]
        source_path = self._candidate_path(str(repo_path))
        if kind == "delete_file":
            return not source_path.exists() and not source_path.is_symlink()
        if kind == "rename_file":
            if not isinstance(target_path, str) or not isinstance(result_sha256, str):
                return False
            target = self._candidate_path(target_path)
            return (
                not source_path.exists()
                and not source_path.is_symlink()
                and self._path_matches_sha256(target, result_sha256)
            )
        if not isinstance(result_sha256, str):
            return False
        matches = self._path_matches_sha256(source_path, result_sha256)
        return matches

    def _overlay_matches_result(self, operation: dict[str, object]) -> bool:
        """Return whether overlay rows match the journaled candidate result."""

        repo_path = operation["repo_path"]
        target_path = operation["target_path"]
        paths = [str(repo_path)]
        if isinstance(target_path, str):
            paths.append(target_path)
        overlay = CandidateOverlay(self.root / "overlay.sqlite")
        try:
            records = overlay.describe_paths(paths)
        finally:
            overlay.close()
        resulting_revision = operation["resulting_candidate_revision"]
        kind = operation["kind"]
        result_sha256 = operation["result_sha256"]
        if kind == "delete_file":
            expected = [("tombstone", None, resulting_revision)]
        elif kind == "rename_file":
            expected = [
                ("tombstone", None, resulting_revision),
                ("content", result_sha256, resulting_revision),
            ]
        else:
            expected = [("content", result_sha256, resulting_revision)]
        actual = [
            (row["state"], row["content_sha256"], row["revision"])
            for row in records
        ]
        return actual == expected

    def _write_result_overlay(self, operation: dict[str, object]) -> None:
        """Reconstruct the missing result overlay from durable candidate files."""

        kind = operation["kind"]
        repo_path = str(operation["repo_path"])
        target_path = operation["target_path"]
        revision = operation["resulting_candidate_revision"]
        if not isinstance(revision, int):
            raise ValueError("candidate recovery resulting revision is invalid")
        overlay = CandidateOverlay(self.root / "overlay.sqlite")
        try:
            if kind == "delete_file":
                overlay.delete(repo_path=repo_path, revision=revision)
            elif kind == "rename_file":
                if not isinstance(target_path, str):
                    raise ValueError("candidate recovery rename target is invalid")
                content = self._safe_text(target_path)
                overlay.rename(
                    source_path=repo_path,
                    target_path=target_path,
                    content=content,
                    revision=revision,
                )
            else:
                content = self._safe_text(repo_path)
                overlay.upsert(
                    repo_path=repo_path,
                    content=content,
                    revision=revision,
                )
        finally:
            overlay.close()

    def _roll_back(self, operation: dict[str, object]) -> None:
        """Restore candidate paths and prior overlay state idempotently."""

        before_paths = operation["before_paths"]
        prior_overlay = operation["prior_overlay"]
        if not isinstance(before_paths, list) or not isinstance(prior_overlay, list):
            raise ValueError("candidate recovery metadata is invalid")
        staged_paths: list[tuple[Path, Path]] = []
        for before_path in before_paths:
            repo_path = str(before_path["repo_path"])
            if not before_path["existed"]:
                continue
            backup_path = self._validated_backup_path(
                str(before_path["backup_relative_path"]),
            )
            candidate_path = self._candidate_path(repo_path)
            staged_path = candidate_path.with_name(
                f".{candidate_path.name}.{operation['operation_id']}.rollback",
            )
            self._write_candidate_file(staged_path, backup_path.read_bytes())
            staged_paths.append((staged_path, candidate_path))
        for staged_path, candidate_path in staged_paths:
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, candidate_path)
        for before_path in before_paths:
            if before_path["existed"]:
                continue
            candidate_path = self._candidate_path(
                str(before_path["repo_path"]),
            )
            if candidate_path.is_dir() or candidate_path.is_symlink():
                raise ValueError("candidate recovery path became unsafe")
            candidate_path.unlink(missing_ok=True)
        for before_path in before_paths:
            candidate_path = self._candidate_path(
                str(before_path["repo_path"]),
            )
            if before_path["existed"]:
                if not self._path_matches_raw_sha256(
                    candidate_path,
                    str(before_path["sha256"]),
                ):
                    raise ValueError("candidate recovery restored identity mismatch")
            elif candidate_path.exists() or candidate_path.is_symlink():
                raise ValueError("candidate recovery absent path was not restored")
        overlay = CandidateOverlay(self.root / "overlay.sqlite")
        try:
            overlay.restore_prior_state(
                records=prior_overlay,
                candidate_source_root=self.root / "source",
            )
            restored_overlay = overlay.describe_paths([
                str(row["repo_path"]) for row in before_paths
            ])
        finally:
            overlay.close()
        if restored_overlay != prior_overlay:
            raise ValueError("candidate recovery prior overlay identity mismatch")
        operation["state"] = "rolled_back"
        self._save()
        self._cleanup_backups(operation)

    def _commit_operation(self, operation: dict[str, object]) -> None:
        """Advance one reconciled operation exactly once and clean backups."""

        expected_revision = operation["resulting_candidate_revision"]
        if not isinstance(expected_revision, int):
            raise ValueError("candidate recovery resulting revision is invalid")
        if expected_revision != self.revision + 1:
            raise ValueError("candidate recovery revision identity mismatch")
        self.revision = expected_revision
        operation["state"] = "committed"
        self._save()
        self._cleanup_backups(operation)

    def _cleanup_backups(self, operation: dict[str, object]) -> None:
        """Best-effort cleanup after terminal journal state is durable."""

        operation_id = operation.get("operation_id")
        if not isinstance(operation_id, str):
            return
        if not (self.root / "recovery").is_dir():
            return
        try:
            recovery_directory = self._recovery_operation_path(operation_id)
        except (FileNotFoundError, ValueError):
            return
        try:
            shutil.rmtree(recovery_directory)
        except FileNotFoundError:
            return
        except OSError:
            return

    def _validated_repo_path(self, value: object) -> str:
        """Return one normalized path confined to the candidate source root."""

        if not isinstance(value, str):
            raise ValueError("candidate repository path is invalid")
        safe_path = normalize_safe_repo_relative_path(value)
        if safe_path is None or safe_path != value:
            raise ValueError("candidate repository path is unsafe")
        return safe_path

    def _candidate_path(self, repo_path: str) -> Path:
        """Return one candidate path after shared symlink confinement."""

        return confined_managed_repo_path(self.root / "source", repo_path)

    def _safe_text(self, repo_path: str) -> str:
        """Read normalized UTF-8 text from one confined candidate file."""

        content = self._safe_file_bytes(self._candidate_path(repo_path))
        if content is None:
            raise ValueError("candidate text path is missing")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("candidate source is not safe text") from exc
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def read_safe_text(self, repo_path: str) -> str | None:
        """Return normalized candidate text or None for an absent path."""

        safe_path = self._validated_repo_path(repo_path)
        content = self._safe_file_bytes(self._candidate_path(safe_path))
        if content is None:
            return None
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("candidate source is not safe text") from exc
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _validated_backup_path(self, relative_path: str) -> Path:
        """Resolve one journal backup beneath the private recovery root."""

        path = Path(relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 3
            or path.parts[0] != "recovery"
        ):
            raise ValueError("candidate recovery backup path is unsafe")
        backup_path = confined_managed_repo_path(
            self.root / "recovery",
            Path(*path.parts[1:]).as_posix(),
        )
        if not backup_path.is_file():
            raise ValueError("candidate recovery backup identity is missing")
        return backup_path

    def _recovery_operation_path(self, operation_id: str) -> Path:
        """Return one confined recovery directory for an operation id."""

        recovery_root = self.root / "recovery"
        if recovery_root.is_symlink():
            raise ValueError("candidate recovery root is unsafe")
        recovery_root.mkdir(parents=True, exist_ok=True)
        return confined_managed_repo_path(recovery_root, operation_id)

    def _validate_current_recovery_path(self, path: Path) -> None:
        """Reject unsafe affected paths before rollback changes any path."""

        if path.is_symlink() or path.is_dir():
            raise ValueError("candidate recovery path became unsafe")
        if path.exists() and not path.is_file():
            raise ValueError("candidate recovery path became unsafe")

    def _safe_file_bytes(self, path: Path) -> bytes | None:
        """Read a regular candidate file while rejecting unsafe path kinds."""

        if path.is_symlink() or path.is_dir():
            raise ValueError("candidate mutation path is unsafe")
        if not path.exists():
            return None
        if not path.is_file():
            raise ValueError("candidate mutation path is not a regular file")
        content = path.read_bytes()
        return content

    def _path_matches_sha256(self, path: Path, expected_sha256: str) -> bool:
        """Return whether safe text has the normalized content identity."""

        try:
            content_bytes = self._safe_file_bytes(path)
        except (OSError, ValueError):
            return False
        if content_bytes is None:
            return False
        try:
            content = content_bytes.decode("utf-8").replace(
                "\r\n",
                "\n",
            ).replace("\r", "\n")
        except UnicodeDecodeError:
            return False
        actual_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return actual_sha256 == expected_sha256

    def _path_matches_raw_sha256(self, path: Path, expected_sha256: str) -> bool:
        """Return whether one restored file matches its exact backup bytes."""

        try:
            content = self._safe_file_bytes(path)
        except (OSError, ValueError):
            return False
        if content is None:
            return False
        actual_sha256 = hashlib.sha256(content).hexdigest()
        return actual_sha256 == expected_sha256

    def _write_candidate_file(self, path: Path, content: bytes) -> None:
        """Write one file atomically with durable content before replacement."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        with temporary_path.open("wb") as file_handle:
            file_handle.write(content)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)

    def _save(self) -> None:
        """Persist the authoritative candidate revision and journal."""

        payload = {"revision": self.revision, "journal": self.journal}
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        self._write_candidate_file(self.root / "state.json", serialized)
