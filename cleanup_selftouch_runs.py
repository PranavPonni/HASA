#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml


VARIANTS = [
    "selftouch_fcn_pos",
    "selftouch_fcn_vel",
    "selftouch_fcn_trq",
    "selftouch_fcn_cmd",
    "selftouch_fcn_posvel",
    "selftouch_fcn_postrq",
    "selftouch_fcn_poscmd",
    "selftouch_fcn_velcmd",
    "selftouch_fcn_veltrq",
    "selftouch_fcn_trqcmd",
    "selftouch_fcn_posveltrq",
    "selftouch_fcn_postrqcmd",
    "selftouch_fcn_poscmdvel",
    "selftouch_fcn_posveltrqcmd",
    "selftouch_fcn_postrqcmd_tplus10",
    "selftouch_fcn_postrqcmd_tplus5",
    "selftouch_fcn_postrqcmd_tplus2",
    "selftouch_fcn_postrqcmd_tminus2",
    "selftouch_fcn_postrqcmd_tminus5",
    "selftouch_fcn_postrqcmd_tminus10",
    "selftouch_fcn_pos_tminus10",
    "selftouch_fcn_pos_tminus5",
    "selftouch_fcn_pos_tminus2",
    "selftouch_fcn_pos_tplus2",
    "selftouch_fcn_pos_tplus5",
    "selftouch_fcn_pos_tplus10",
]


def load_entity(root):
    for variant in VARIANTS:
        path = root / "parameter" / variant / "parameter_base" / "parameter_base.yaml"
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text())
        entity = (
            data.get("Sweep", {}).get("wandb_entity")
            or data.get("Train", {}).get("wandb_entity")
        )
        if entity:
            return entity
    return os.environ.get("WANDB_ENTITY")


def remove_path(path, dry_run=False):
    if not path.exists() and not path.is_symlink():
        return False
    print(f"{'would remove' if dry_run else 'removing'} {path}")
    if dry_run:
        return True
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
    return True


def cleanup_generated_dirs(root, dry_run=False):
    removed = 0
    for top_name in ("parameter", "model_weight"):
        for variant in VARIANTS:
            base = root / top_name / variant
            if not base.is_dir():
                continue
            for child in sorted(base.iterdir()):
                if child.name == "parameter_base":
                    continue
                removed += int(remove_path(child, dry_run=dry_run))
    return removed


def marker_strings():
    markers = set(VARIANTS)
    markers.update(f"parameter/{variant}/" for variant in VARIANTS)
    markers.update(f"model_weight/{variant}/" for variant in VARIANTS)
    return tuple(sorted(markers))


def file_contains_marker(path, markers):
    try:
        if path.stat().st_size > 2_000_000:
            return False
        text = path.read_text(errors="ignore")
    except OSError:
        return False
    return any(marker in text for marker in markers)


def tree_contains_marker(path, markers):
    if path.is_symlink():
        return False
    if path.is_file():
        return file_contains_marker(path, markers)
    for child in path.rglob("*"):
        if child.is_file() and file_contains_marker(child, markers):
            return True
    return False


def cleanup_local_wandb(root, dry_run=False):
    wandb_root = root / "wandb"
    if not wandb_root.exists():
        return 0
    markers = marker_strings()
    removed = 0
    for child in sorted(wandb_root.iterdir()):
        if child.name.startswith(("run-", "offline-run-", "sweep-")):
            if tree_contains_marker(child, markers):
                removed += int(remove_path(child, dry_run=dry_run))
    for name in ("latest-run", "debug.log", "debug-internal.log"):
        path = wandb_root / name
        if path.is_symlink() and not path.exists():
            removed += int(remove_path(path, dry_run=dry_run))
    return removed


def cleanup_logs(root, dry_run=False):
    return int(remove_path(root / "logs" / "selftouch_fcn_variants", dry_run=dry_run))


def delete_wandb_project(api, entity, project):
    project_obj = api.project(project, entity=entity)
    if hasattr(project_obj, "_load") and not project_obj._attrs.get("id"):
        project_obj._load()
    project_id = project_obj._attrs.get("id")
    if not project_id:
        raise RuntimeError(f"Could not resolve W&B project id for {entity}/{project}")
    mutation = """
        mutation deleteProject($id: String!) {
          deleteModel(input: {id: $id}) {
            success
            __typename
          }
        }
        """
    variables = {"id": project_id}
    service_api = getattr(api, "_service_api", None)
    if service_api is not None and hasattr(service_api, "execute_graphql"):
        # W&B >= 0.28 accepts raw GraphQL strings through its service API and
        # no longer ships the old standalone wandb_gql parser.
        result = service_api.execute_graphql(mutation, variables=variables)
    else:
        # Compatibility with older W&B clients whose RetryingClient expects a
        # parsed GraphQL document.
        from wandb_gql import gql

        result = api.client.execute(gql(mutation), variable_values=variables)
    success = result.get("deleteModel", {}).get("success")
    if not success:
        raise RuntimeError(f"W&B project delete failed for {entity}/{project}: {result}")


def cleanup_wandb_online(entity, delete_projects=False, delete_artifacts=False, dry_run=False):
    import wandb

    api = wandb.Api()
    deleted_runs = 0
    deleted_projects = 0
    failures = []
    for project in VARIANTS:
        path = f"{entity}/{project}"
        if delete_projects:
            print(f"{'would delete' if dry_run else 'deleting'} W&B project {path}")
            if not dry_run:
                try:
                    delete_wandb_project(api, entity, project)
                except Exception as exc:
                    failures.append((path, str(exc)))
                    print(f"  FAILED {path}: {exc}", file=sys.stderr)
                    continue
            deleted_projects += 1
            continue

        print(f"checking W&B runs in {path}")
        try:
            runs = list(api.runs(path))
        except Exception as exc:
            failures.append((path, str(exc)))
            print(f"  skipped {path}: {exc}", file=sys.stderr)
            continue
        for run in runs:
            print(f"  {'would delete' if dry_run else 'deleting'} run {run.name} ({run.id})")
            if not dry_run:
                run.delete(delete_artifacts=delete_artifacts)
            deleted_runs += 1
    return deleted_runs, deleted_projects, failures


def main():
    parser = argparse.ArgumentParser(description="Delete local and optional W&B selftouch FCN run artifacts.")
    parser.add_argument("--root", default=".", help="motionlearning repo root")
    parser.add_argument("--wandb", action="store_true", help="also delete online W&B runs/projects")
    parser.add_argument("--delete-projects", action="store_true", help="with --wandb, delete the configured W&B projects themselves")
    parser.add_argument("--delete-artifacts", action="store_true", help="with --wandb run deletion, also delete run artifacts")
    parser.add_argument("--entity", default=None, help="W&B entity; defaults to parameter_base.yaml or WANDB_ENTITY")
    parser.add_argument("--dry-run", action="store_true", help="print what would be deleted")
    parser.add_argument("--yes", action="store_true", help="required for real deletion")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not args.dry_run and not args.yes:
        raise SystemExit("Refusing to delete without --yes. Use --dry-run to preview.")

    entity = args.entity or load_entity(root)
    local_count = 0
    local_count += cleanup_generated_dirs(root, dry_run=args.dry_run)
    local_count += cleanup_local_wandb(root, dry_run=args.dry_run)
    local_count += cleanup_logs(root, dry_run=args.dry_run)
    print(f"local_removed={local_count}")

    if args.wandb:
        if not entity:
            raise SystemExit("W&B entity not found; pass --entity.")
        runs, projects, failures = cleanup_wandb_online(
            entity,
            delete_projects=args.delete_projects,
            delete_artifacts=args.delete_artifacts,
            dry_run=args.dry_run,
        )
        print(f"wandb_deleted_runs={runs} wandb_deleted_projects={projects}")
        if failures:
            print(f"wandb_failures={len(failures)}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
