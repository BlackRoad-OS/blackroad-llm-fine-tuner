"""LLM fine-tuning job manager with support for multiple model families."""

import sqlite3
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import argparse
import uuid

# Database initialization
DB_PATH = Path.home() / ".blackroad" / "finetune.db"

SUPPORTED_MODELS = ["llama3.2", "qwen2.5", "mistral", "deepseek-r1", "phi-3"]

JOB_STATUSES = ["queued", "running", "completed", "failed"]


@dataclass
class FinetuneJob:
    """Represents a fine-tuning job."""
    id: str
    base_model: str
    dataset_path: str
    status: str
    epochs: int
    lr: float
    batch_size: int
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    val_loss: Optional[float] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


class LLMFineTuner:
    """Manager for LLM fine-tuning jobs."""

    def __init__(self, db_path: Path = DB_PATH):
        """Initialize the fine-tuner with SQLite database."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    base_model TEXT NOT NULL,
                    dataset_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    epochs INTEGER NOT NULL,
                    lr REAL NOT NULL,
                    batch_size INTEGER NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    output_path TEXT,
                    val_loss REAL,
                    error TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def create_job(
        self,
        base_model: str,
        dataset_path: str,
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 4,
    ) -> str:
        """Queue a new fine-tuning job."""
        if base_model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model. Supported: {SUPPORTED_MODELS}")

        job_id = str(uuid.uuid4())[:8]
        job = FinetuneJob(
            id=job_id,
            base_model=base_model,
            dataset_path=dataset_path,
            status="queued",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs VALUES (
                    :id, :base_model, :dataset_path, :status, :epochs, :lr,
                    :batch_size, :started_at, :completed_at, :output_path,
                    :val_loss, :error, :created_at
                )
                """,
                job.to_dict(),
            )
            conn.commit()

        return job_id

    def start_job(self, job_id: str):
        """Mark job as running."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs SET status = ?, started_at = ?
                WHERE id = ?
                """,
                ("running", datetime.utcnow().isoformat(), job_id),
            )
            conn.commit()

    def complete_job(self, job_id: str, val_loss: float, output_path: str):
        """Mark job as completed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs SET status = ?, completed_at = ?, val_loss = ?, output_path = ?
                WHERE id = ?
                """,
                (
                    "completed",
                    datetime.utcnow().isoformat(),
                    val_loss,
                    output_path,
                    job_id,
                ),
            )
            conn.commit()

    def fail_job(self, job_id: str, error: str):
        """Mark job as failed with error message."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs SET status = ?, error = ?, completed_at = ?
                WHERE id = ?
                """,
                ("failed", error, datetime.utcnow().isoformat(), job_id),
            )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[FinetuneJob]:
        """Retrieve job details."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()

        if row:
            return FinetuneJob(**dict(row))
        return None

    def list_jobs(self, status: Optional[str] = None) -> List[FinetuneJob]:
        """List jobs, optionally filtered by status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if status:
                cursor = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                )
            else:
                cursor = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            rows = cursor.fetchall()

        return [FinetuneJob(**dict(row)) for row in rows]

    def compare_jobs(self, job_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple jobs."""
        jobs = [self.get_job(jid) for jid in job_ids if self.get_job(jid)]

        comparison = {
            "job_count": len(jobs),
            "jobs": [
                {
                    "id": j.id,
                    "model": j.base_model,
                    "status": j.status,
                    "epochs": j.epochs,
                    "lr": j.lr,
                    "batch_size": j.batch_size,
                    "val_loss": j.val_loss,
                }
                for j in jobs
            ],
        }

        completed = [j for j in jobs if j.status == "completed"]
        if completed:
            losses = [j.val_loss for j in completed if j.val_loss is not None]
            if losses:
                comparison["best_val_loss"] = min(losses)
                comparison["worst_val_loss"] = max(losses)
                comparison["avg_val_loss"] = sum(losses) / len(losses)

        return comparison

    def estimate_cost(
        self, dataset_size_mb: float, epochs: int, gpu_type: str = "A100"
    ) -> Dict[str, Any]:
        """Estimate training cost based on dataset size and parameters."""
        # Rough estimates (USD/hour for different GPUs)
        gpu_costs = {
            "A100": 2.5,
            "H100": 3.5,
            "V100": 0.95,
            "T4": 0.35,
        }

        hourly_rate = gpu_costs.get(gpu_type, 2.5)

        # Very rough estimate: 1GB = 0.5 hours per epoch
        est_hours_per_epoch = max(0.25, (dataset_size_mb / 1024) * 0.5)
        total_hours = est_hours_per_epoch * epochs
        total_cost = total_hours * hourly_rate

        return {
            "gpu_type": gpu_type,
            "dataset_size_mb": dataset_size_mb,
            "epochs": epochs,
            "est_hours_per_epoch": est_hours_per_epoch,
            "total_hours": total_hours,
            "hourly_rate_usd": hourly_rate,
            "estimated_cost_usd": round(total_cost, 2),
        }


def main():
    """CLI interface for fine-tuner."""
    parser = argparse.ArgumentParser(description="LLM Fine-Tuner Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List fine-tuning jobs")
    list_parser.add_argument(
        "--status", help="Filter by status (queued/running/completed/failed)"
    )

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new fine-tuning job")
    create_parser.add_argument("model", help="Base model to fine-tune")
    create_parser.add_argument("dataset", help="Path to training dataset (JSONL)")
    create_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    create_parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    create_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare jobs")
    compare_parser.add_argument("job_ids", nargs="+", help="Job IDs to compare")

    # Cost estimate command
    cost_parser = subparsers.add_parser("cost", help="Estimate training cost")
    cost_parser.add_argument("dataset_size_mb", type=float, help="Dataset size in MB")
    cost_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    cost_parser.add_argument(
        "--gpu", default="A100", help="GPU type (A100/H100/V100/T4)"
    )

    args = parser.parse_args()
    ft = LLMFineTuner()

    if args.command == "list":
        jobs = ft.list_jobs(status=args.status)
        if not jobs:
            print("No jobs found.")
            return
        print(f"{'ID':<10} {'Model':<15} {'Status':<12} {'Epochs':<8} {'Val Loss':<12}")
        print("-" * 60)
        for job in jobs:
            val_loss_str = f"{job.val_loss:.4f}" if job.val_loss else "N/A"
            print(
                f"{job.id:<10} {job.base_model:<15} {job.status:<12} {job.epochs:<8} {val_loss_str:<12}"
            )

    elif args.command == "create":
        job_id = ft.create_job(
            args.model,
            args.dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        print(f"Created job {job_id} with model {args.model}")

    elif args.command == "compare":
        result = ft.compare_jobs(args.job_ids)
        print(json.dumps(result, indent=2))

    elif args.command == "cost":
        result = ft.estimate_cost(
            args.dataset_size_mb, args.epochs, gpu_type=args.gpu
        )
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
