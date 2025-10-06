import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

# apply library caps
try:
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

from celery import Celery
from dotenv import load_dotenv
import multiprocessing

multiprocessing.set_start_method("spawn")

load_dotenv(".env")

celery_app = Celery(
    "celery_app",
    broker=os.environ.get("REDIS_HOST", "redis://localhost:6379/0"),
    backend=os.environ.get("REDIS_HOST", "redis://localhost:6379/0"),
    include=["marker_api.celery_tasks"],
)


@celery_app.task(name="celery.ping")
def ping():
    print("Ping task received!")  # or use a logger
    return "pong"
