import os
os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "eager")

# Patch Surya's MBART attention map so 'sdpa' falls back to 'eager'
import surya.model.ordering.decoder as sdec  # noqa: E402
if "sdpa" not in sdec.MBART_ATTENTION_CLASSES:
    sdec.MBART_ATTENTION_CLASSES["sdpa"] = sdec.MBART_ATTENTION_CLASSES["eager"]

import io
import logging
from threading import Lock
from typing import Any, Dict, Tuple

from celery import Task
from celery.signals import worker_ready

from marker_api.celery_worker import celery_app
from marker_api.utils import process_image_to_base64
from marker.convert import convert_single_pdf
from marker.models import load_all_models

logger = logging.getLogger(__name__)

# Loaded once per worker process
model_list: Tuple[Any, ...] | None = None
_model_lock = Lock()


def get_models() -> Tuple[Any, ...]:
    """Lazy, process/thread-safe model loader."""
    global model_list
    if model_list is None:
        with _model_lock:
            if model_list is None:
                logger.info("Loading models...")
                model_list = load_all_models()
                logger.info("Models loaded.")
    return model_list


@worker_ready.connect
def initialize_models(**_: Dict[str, Any]) -> None:
    """Warm-up at worker start. If it fails, lazy-load will retry on first task."""
    try:
        get_models()
    except Exception:
        logger.exception("Model warm-up failed; will retry on first task.")


class PDFConversionTask(Task):
    abstract = True

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


@celery_app.task(ignore_result=False, bind=True, base=PDFConversionTask, name="convert_pdf")
def convert_pdf_to_markdown(self, filename: str, pdf_content: bytes) -> Dict[str, Any]:
    pdf_file = io.BytesIO(pdf_content)
    models = get_models()
    markdown_text, images, metadata = convert_single_pdf(pdf_file, models)

    image_data: Dict[str, str] = {}
    for img_filename, image in images.items():
        logger.debug("Processing image %s", img_filename)
        image_data[img_filename] = process_image_to_base64(image, img_filename)

    return {
        "filename": filename,
        "markdown": markdown_text,
        "metadata": metadata,
        "images": image_data,
        "status": "ok",
    }


@celery_app.task(ignore_result=False, bind=True, base=PDFConversionTask, name="process_batch")
def process_batch(self, batch_data: list[tuple[str, bytes]]) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    total = len(batch_data)
    models = get_models()  # load once for the batch

    for i, (filename, pdf_content) in enumerate(batch_data, start=1):
        try:
            pdf_file = io.BytesIO(pdf_content)
            markdown_text, images, metadata = convert_single_pdf(pdf_file, models)

            image_data: Dict[str, str] = {}
            for img_filename, image in images.items():
                image_data[img_filename] = process_image_to_base64(image, img_filename)

            results.append({
                "filename": filename,
                "markdown": markdown_text,
                "metadata": metadata,
                "images": image_data,
                "status": "ok",
            })
        except Exception as e:
            logger.exception("Error processing %s", filename)
            results.append({"filename": filename, "status": "Error", "error": str(e)})

        # progress meta for polling UIs
        try:
            self.update_state(state="PROGRESS", meta={"current": i, "total": total})
        except Exception:
            pass

    return results
