#!/usr/bin/env python3
"""Distributed Job Server - Route processing to desktop/laptop workers.

Architecture:
- Server runs on laptop (has USB capture card)
- Workers run on desktop (GPU) and/or laptop (CPU)
- Jobs are queued and distributed based on worker capability
- Results collected back to laptop

Network:
- Laptop (omarchy-1): 100.103.8.87
- Desktop (omarchy):  100.92.71.80

Usage:
    # Start server on laptop
    python job_server.py --server --port 8780

    # Start worker on desktop
    python job_server.py --worker --server-url http://100.103.8.87:8780 --device gpu

    # Start worker on laptop (CPU fallback)
    python job_server.py --worker --server-url http://localhost:8780 --device cpu

    # Submit batch job
    python job_server.py --submit --job-type l3_batch --input-dir /path/to/keyframes
"""

import os
import sys
import json
import time
import queue
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import shutil

# FastAPI for server
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    import uvicorn
    import httpx
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("Warning: FastAPI not installed. Server mode unavailable.")

# Paths
UI_AGENT_DIR = Path(__file__).parent.parent.parent
JOBS_DIR = UI_AGENT_DIR / "jobs"

# Network
LAPTOP_IP = "100.103.8.87"
DESKTOP_IP = "100.92.71.80"
DEFAULT_PORT = 8780


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    L3_SINGLE = "l3_single"      # Analyze single keyframe
    L3_BATCH = "l3_batch"        # Analyze batch of keyframes
    L2_SUMMARY = "l2_summary"    # Summarize L3 results
    CAPTURE = "capture"          # Video capture (laptop only)


class DeviceType(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    ANY = "any"


@dataclass
class Job:
    job_id: str
    job_type: JobType
    status: JobStatus
    preferred_device: DeviceType
    input_data: dict
    output_data: Optional[dict] = None
    worker_id: Optional[str] = None
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self):
        return asdict(self)


@dataclass
class Worker:
    worker_id: str
    device_type: DeviceType
    hostname: str
    ip_address: str
    status: str = "idle"
    current_job: Optional[str] = None
    last_heartbeat: str = ""
    jobs_completed: int = 0


# ============================================================================
# SERVER
# ============================================================================

if HAS_FASTAPI:
    app = FastAPI(title="UI-Agent Job Server")

    # In-memory state (could be Redis/SQLite for persistence)
    jobs: Dict[str, Job] = {}
    workers: Dict[str, Worker] = {}
    job_queue: queue.Queue = queue.Queue()

    @app.get("/")
    async def root():
        return {
            "service": "UI-Agent Job Server",
            "jobs_total": len(jobs),
            "jobs_pending": sum(1 for j in jobs.values() if j.status == JobStatus.PENDING),
            "workers": len(workers),
            "workers_idle": sum(1 for w in workers.values() if w.status == "idle")
        }

    @app.post("/jobs")
    async def create_job(job_type: str, preferred_device: str = "any", input_data: dict = {}):
        """Create a new job."""
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(jobs):04d}"
        job = Job(
            job_id=job_id,
            job_type=JobType(job_type),
            status=JobStatus.PENDING,
            preferred_device=DeviceType(preferred_device),
            input_data=input_data
        )
        jobs[job_id] = job
        job_queue.put(job_id)

        return {"job_id": job_id, "status": "created"}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        """Get job status."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return jobs[job_id].to_dict()

    @app.get("/jobs")
    async def list_jobs(status: Optional[str] = None, limit: int = 100):
        """List jobs."""
        result = list(jobs.values())
        if status:
            result = [j for j in result if j.status == status]
        return {"jobs": [j.to_dict() for j in result[-limit:]]}

    @app.post("/workers/register")
    async def register_worker(worker_id: str, device_type: str, hostname: str, ip_address: str):
        """Register a worker."""
        worker = Worker(
            worker_id=worker_id,
            device_type=DeviceType(device_type),
            hostname=hostname,
            ip_address=ip_address,
            last_heartbeat=datetime.now().isoformat()
        )
        workers[worker_id] = worker
        return {"status": "registered", "worker_id": worker_id}

    @app.post("/workers/{worker_id}/heartbeat")
    async def worker_heartbeat(worker_id: str, status: str = "idle", current_job: Optional[str] = None):
        """Worker heartbeat."""
        if worker_id not in workers:
            raise HTTPException(status_code=404, detail="Worker not found")

        workers[worker_id].last_heartbeat = datetime.now().isoformat()
        workers[worker_id].status = status
        workers[worker_id].current_job = current_job
        return {"status": "ok"}

    @app.get("/workers/{worker_id}/next-job")
    async def get_next_job(worker_id: str):
        """Get next job for worker."""
        if worker_id not in workers:
            raise HTTPException(status_code=404, detail="Worker not found")

        worker = workers[worker_id]

        # Find matching job
        for job_id, job in jobs.items():
            if job.status != JobStatus.PENDING:
                continue

            # Check device preference
            if job.preferred_device != DeviceType.ANY:
                if job.preferred_device != worker.device_type:
                    continue

            # Assign job
            job.status = JobStatus.RUNNING
            job.worker_id = worker_id
            job.started_at = datetime.now().isoformat()
            worker.status = "busy"
            worker.current_job = job_id

            return job.to_dict()

        return {"job": None}

    @app.post("/workers/{worker_id}/complete-job")
    async def complete_job(worker_id: str, job_id: str, output_data: dict = {}, error: Optional[str] = None):
        """Mark job as completed."""
        if worker_id not in workers:
            raise HTTPException(status_code=404, detail="Worker not found")
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        worker = workers[worker_id]

        if error:
            job.status = JobStatus.FAILED
            job.error = error
        else:
            job.status = JobStatus.COMPLETED
            job.output_data = output_data

        job.completed_at = datetime.now().isoformat()
        worker.status = "idle"
        worker.current_job = None
        worker.jobs_completed += 1

        return {"status": "ok"}

    @app.post("/submit-batch")
    async def submit_batch(keyframe_paths: List[str], preferred_device: str = "gpu"):
        """Submit a batch of keyframes for L3 analysis."""
        job_ids = []
        for path in keyframe_paths:
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(jobs):04d}"
            job = Job(
                job_id=job_id,
                job_type=JobType.L3_SINGLE,
                status=JobStatus.PENDING,
                preferred_device=DeviceType(preferred_device),
                input_data={"keyframe_path": path}
            )
            jobs[job_id] = job
            job_ids.append(job_id)

        return {"jobs_created": len(job_ids), "job_ids": job_ids}


# ============================================================================
# WORKER
# ============================================================================

class JobWorker:
    """Worker that processes jobs from the server."""

    def __init__(self, server_url: str, device_type: str, worker_id: str = None):
        self.server_url = server_url.rstrip("/")
        self.device_type = device_type
        self.worker_id = worker_id or f"worker_{device_type}_{datetime.now().strftime('%H%M%S')}"
        self.hostname = os.uname().nodename
        self.running = False

    def register(self):
        """Register with server."""
        import socket
        ip = socket.gethostbyname(socket.gethostname())

        response = httpx.post(f"{self.server_url}/workers/register", params={
            "worker_id": self.worker_id,
            "device_type": self.device_type,
            "hostname": self.hostname,
            "ip_address": ip
        })
        response.raise_for_status()
        print(f"Registered as {self.worker_id} ({self.device_type})")

    def heartbeat(self, status: str = "idle", current_job: str = None):
        """Send heartbeat."""
        try:
            httpx.post(f"{self.server_url}/workers/{self.worker_id}/heartbeat", params={
                "status": status,
                "current_job": current_job
            }, timeout=5)
        except Exception as e:
            print(f"Heartbeat failed: {e}")

    def get_next_job(self) -> Optional[dict]:
        """Get next job from server."""
        try:
            response = httpx.get(f"{self.server_url}/workers/{self.worker_id}/next-job", timeout=10)
            data = response.json()
            if data.get("job"):
                return data
            return data if data.get("job_id") else None
        except Exception as e:
            print(f"Error getting job: {e}")
            return None

    def complete_job(self, job_id: str, output_data: dict = None, error: str = None):
        """Mark job as completed."""
        try:
            httpx.post(f"{self.server_url}/workers/{self.worker_id}/complete-job", params={
                "job_id": job_id,
                "error": error
            }, json={"output_data": output_data or {}}, timeout=30)
        except Exception as e:
            print(f"Error completing job: {e}")

    def process_l3_job(self, job: dict) -> dict:
        """Process L3 analysis job."""
        keyframe_path = job["input_data"]["keyframe_path"]

        prompt = f'''Read the image file at {keyframe_path} and describe in 2-3 detailed sentences.

Include:
1. WHAT application/window is visible (be specific - app name, window title)
2. WHAT the user is doing (typing, clicking, scrolling, reading, waiting)
3. WHAT data/content is visible (file names, form fields, text content, errors)
4. WHAT UI state (dialogs, menus, loading indicators, cursor position)

Be specific. Include readable text. Reply with ONLY the description.'''

        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(UI_AGENT_DIR)
            )

            if result.returncode == 0 and result.stdout.strip():
                return {
                    "keyframe": Path(keyframe_path).name,
                    "description": result.stdout.strip(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise Exception(f"Claude error: {result.stderr}")

        except Exception as e:
            raise Exception(f"L3 analysis failed: {e}")

    def run(self):
        """Main worker loop."""
        self.register()
        self.running = True

        print(f"Worker {self.worker_id} starting...")
        print(f"Server: {self.server_url}")
        print(f"Device: {self.device_type}")

        last_heartbeat = 0

        while self.running:
            try:
                # Heartbeat every 10 seconds
                if time.time() - last_heartbeat > 10:
                    self.heartbeat("idle")
                    last_heartbeat = time.time()

                # Get next job
                job = self.get_next_job()

                if job and job.get("job_id"):
                    job_id = job["job_id"]
                    job_type = job["job_type"]

                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing {job_id} ({job_type})")
                    self.heartbeat("busy", job_id)

                    try:
                        if job_type == "l3_single":
                            output = self.process_l3_job(job)
                            self.complete_job(job_id, output)
                            print(f"  ✓ Completed: {output.get('description', '')[:60]}...")
                        else:
                            self.complete_job(job_id, error=f"Unknown job type: {job_type}")
                            print(f"  ✗ Unknown job type: {job_type}")

                    except Exception as e:
                        self.complete_job(job_id, error=str(e))
                        print(f"  ✗ Error: {e}")

                else:
                    # No jobs available, wait
                    time.sleep(2)

            except KeyboardInterrupt:
                print("\nShutting down worker...")
                self.running = False
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(5)


# ============================================================================
# CLI
# ============================================================================

def run_server(port: int = DEFAULT_PORT):
    """Run the job server."""
    if not HAS_FASTAPI:
        print("FastAPI not installed. Run: pip install fastapi uvicorn httpx")
        return

    print(f"Starting job server on port {port}")
    print(f"Workers can connect to: http://{LAPTOP_IP}:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


def run_worker(server_url: str, device_type: str):
    """Run a worker."""
    worker = JobWorker(server_url, device_type)
    worker.run()


def submit_keyframes(server_url: str, keyframe_dir: str, device: str = "gpu"):
    """Submit keyframes for processing."""
    keyframe_dir = Path(keyframe_dir)
    keyframes = list(keyframe_dir.glob("*.png"))

    print(f"Found {len(keyframes)} keyframes")

    response = httpx.post(f"{server_url}/submit-batch", json={
        "keyframe_paths": [str(k) for k in keyframes],
        "preferred_device": device
    })

    data = response.json()
    print(f"Created {data['jobs_created']} jobs")
    return data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Distributed Job Server")
    parser.add_argument("--server", action="store_true", help="Run as server")
    parser.add_argument("--worker", action="store_true", help="Run as worker")
    parser.add_argument("--submit", action="store_true", help="Submit jobs")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--server-url", type=str, help="Server URL for worker")
    parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default="cpu", help="Worker device type")
    parser.add_argument("--input-dir", type=str, help="Input directory for submit")

    args = parser.parse_args()

    if args.server:
        run_server(args.port)
    elif args.worker:
        if not args.server_url:
            print("--server-url required for worker mode")
            return
        run_worker(args.server_url, args.device)
    elif args.submit:
        if not args.server_url or not args.input_dir:
            print("--server-url and --input-dir required for submit")
            return
        submit_keyframes(args.server_url, args.input_dir, args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
