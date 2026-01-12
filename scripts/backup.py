#!/usr/bin/env python3
"""
Backup Script
Creates hourly backups of trading data, logs, and model checkpoints
"""

import argparse
import json
import os
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path


def create_backup(
    backup_dir: str = "data/backups",
    include_models: bool = True,
    include_logs: bool = True,
    compress: bool = True
) -> str:
    """
    Create a backup of all important data
    
    Returns:
        Path to the created backup
    """
    project_root = Path(__file__).parent.parent
    backup_base = project_root / backup_dir
    
    # Create timestamp-based backup directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_base / timestamp
    backup_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating backup at: {backup_path}")
    
    # Directories to backup
    backup_items = [
        ("config", project_root / "config"),
        ("data/trading", project_root / "data" / "trading"),
        ("data/backtests", project_root / "data" / "backtests"),
    ]
    
    if include_models:
        backup_items.append(("models", project_root / "models"))
    
    if include_logs:
        backup_items.append(("logs", project_root / "logs"))
    
    # Copy files
    for name, source in backup_items:
        if source.exists():
            dest = backup_path / name
            if source.is_dir():
                shutil.copytree(source, dest, dirs_exist_ok=True)
                print(f"  ✓ Backed up {name}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                print(f"  ✓ Backed up {name}")
        else:
            print(f"  ⚠ Skipped {name} (not found)")
    
    # Create backup metadata
    metadata = {
        "timestamp": timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": [name for name, _ in backup_items],
        "include_models": include_models,
        "include_logs": include_logs
    }
    
    with open(backup_path / "backup_info.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Compress if requested
    if compress:
        archive_path = backup_base / f"backup_{timestamp}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_path, arcname=timestamp)
        
        # Remove uncompressed directory
        shutil.rmtree(backup_path)
        print(f"\n✓ Created compressed backup: {archive_path}")
        return str(archive_path)
    
    print(f"\n✓ Backup complete: {backup_path}")
    return str(backup_path)


def cleanup_old_backups(
    backup_dir: str = "data/backups",
    keep_days: int = 7,
    keep_count: int = 24
) -> int:
    """
    Remove old backups
    
    Args:
        backup_dir: Directory containing backups
        keep_days: Keep backups from last N days
        keep_count: Minimum number of backups to keep
    
    Returns:
        Number of backups removed
    """
    project_root = Path(__file__).parent.parent
    backup_base = project_root / backup_dir
    
    if not backup_base.exists():
        return 0
    
    # Find all backup files/directories
    backups = []
    
    for item in backup_base.iterdir():
        if item.name.startswith("backup_") or item.name.isdigit() or "_" in item.name[:8]:
            mtime = item.stat().st_mtime
            backups.append((item, mtime))
    
    # Sort by modification time (newest first)
    backups.sort(key=lambda x: x[1], reverse=True)
    
    # Determine cutoff
    cutoff_time = time.time() - (keep_days * 86400)
    
    removed = 0
    for i, (backup, mtime) in enumerate(backups):
        # Keep minimum count
        if i < keep_count:
            continue
        
        # Keep if within time window
        if mtime > cutoff_time:
            continue
        
        # Remove old backup
        try:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
            removed += 1
            print(f"  Removed: {backup.name}")
        except Exception as e:
            print(f"  ⚠ Failed to remove {backup.name}: {e}")
    
    return removed


def list_backups(backup_dir: str = "data/backups") -> None:
    """List all available backups"""
    project_root = Path(__file__).parent.parent
    backup_base = project_root / backup_dir
    
    if not backup_base.exists():
        print("No backups found")
        return
    
    backups = []
    for item in backup_base.iterdir():
        if item.name.startswith("backup_") or item.name.isdigit():
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) if item.is_dir() else item.stat().st_size
            mtime = item.stat().st_mtime
            backups.append((item.name, size, mtime))
    
    if not backups:
        print("No backups found")
        return
    
    # Sort by time (newest first)
    backups.sort(key=lambda x: x[2], reverse=True)
    
    print("\nAvailable Backups:")
    print("-" * 60)
    print(f"{'Name':<35} {'Size':>10} {'Age':>12}")
    print("-" * 60)
    
    for name, size, mtime in backups:
        age_seconds = time.time() - mtime
        
        if age_seconds < 3600:
            age_str = f"{int(age_seconds / 60)}m ago"
        elif age_seconds < 86400:
            age_str = f"{int(age_seconds / 3600)}h ago"
        else:
            age_str = f"{int(age_seconds / 86400)}d ago"
        
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        
        print(f"{name:<35} {size_str:>10} {age_str:>12}")
    
    print("-" * 60)
    print(f"Total: {len(backups)} backups")


def restore_backup(backup_name: str, backup_dir: str = "data/backups") -> bool:
    """
    Restore from a backup
    
    Args:
        backup_name: Name of backup to restore
        backup_dir: Directory containing backups
    
    Returns:
        True if successful
    """
    project_root = Path(__file__).parent.parent
    backup_base = project_root / backup_dir
    backup_path = backup_base / backup_name
    
    # Check for compressed backup
    if not backup_path.exists():
        archive_path = backup_base / f"{backup_name}.tar.gz"
        if archive_path.exists():
            print(f"Extracting {archive_path}...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(backup_base)
            backup_path = backup_base / backup_name.replace("backup_", "")
    
    if not backup_path.exists():
        print(f"Backup not found: {backup_name}")
        return False
    
    print(f"Restoring from: {backup_path}")
    
    # Restore each directory
    for item in backup_path.iterdir():
        if item.name == "backup_info.json":
            continue
        
        dest = project_root / item.name
        
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            print(f"  ✓ Restored {item.name}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            print(f"  ✓ Restored {item.name}")
    
    print("\n✓ Restore complete")
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MLA HFT Backup Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create backup command
    create_parser = subparsers.add_parser("create", help="Create a new backup")
    create_parser.add_argument("--no-models", action="store_true", help="Exclude model files")
    create_parser.add_argument("--no-logs", action="store_true", help="Exclude log files")
    create_parser.add_argument("--no-compress", action="store_true", help="Don't compress backup")
    
    # List backups command
    subparsers.add_parser("list", help="List available backups")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old backups")
    cleanup_parser.add_argument("--keep-days", type=int, default=7, help="Keep backups from last N days")
    cleanup_parser.add_argument("--keep-count", type=int, default=24, help="Minimum backups to keep")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("backup_name", help="Name of backup to restore")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_backup(
            include_models=not args.no_models,
            include_logs=not args.no_logs,
            compress=not args.no_compress
        )
    elif args.command == "list":
        list_backups()
    elif args.command == "cleanup":
        removed = cleanup_old_backups(
            keep_days=args.keep_days,
            keep_count=args.keep_count
        )
        print(f"\nRemoved {removed} old backup(s)")
    elif args.command == "restore":
        restore_backup(args.backup_name)
    else:
        # Default: create backup
        create_backup()


if __name__ == "__main__":
    main()
