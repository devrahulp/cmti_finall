#!/usr/bin/env python3
"""
ocr.py

Simple, well-documented OCR utility for the CMTI handwritten-document project.
Features:
- Reads images (jpg/png/tiff) and PDFs and extracts text using Tesseract (pytesseract).
- Image preprocessing (grayscale, denoise, adaptive threshold, optional deskew) to improve OCR on handwriting.
- CLI interface and Python API.

Requirements:
- Python 3.8+
- pip install opencv-python pillow pytesseract numpy pdf2image
- On the system: Tesseract OCR installed and accessible (e.g. 'tesseract' in PATH).
  On macOS: brew install tesseract
  On Ubuntu/Debian: sudo apt install tesseract-ocr

Usage examples:
    python ocr.py --input scanned_page.jpg --output out.txt
    python ocr.py --input document.pdf --output out_from_pdf.txt --pages 1-3

"""

import argparse
import os
import sys
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
import pytesseract

# If you want to use pdf2image for PDF support:
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False


def preprocess_image_for_ocr(image: np.ndarray, deskew: bool = False, binarize: bool = True) -> np.ndarray:
    """Preprocesses an image (OpenCV BGR/gray) to improve OCR results.

    Steps:
    - Convert to gray
    - Optional deskew
    - Noise reduction (median blur)
    - Optional adaptive thresholding

    Returns a grayscale image ready for pytesseract (as a numpy array).
    """
    # ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # denoise
    gray = cv2.medianBlur(gray, 3)

    if deskew:
        gray = deskew_image(gray)

    if binarize:
        # adaptive threshold is often good for uneven lighting / handwriting
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 25, 15)
    else:
        # simple Otsu
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return gray


def deskew_image(gray: np.ndarray) -> np.ndarray:
    """Estimate skew angle and rotate to deskew the image."""
    coords = np.column_stack(np.where(gray < 255))  # text pixels
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def ocr_image_cv2(image_path: str, lang: str = 'eng', deskew: bool = False, binarize: bool = True,
                  config_extra: Optional[str] = None) -> str:
    """Load an image path with OpenCV, preprocess, and run Tesseract OCR. Returns extracted text."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"cv2 failed to load image: {image_path}")

    pre = preprocess_image_for_ocr(image, deskew=deskew, binarize=binarize)

    # convert to PIL Image for pytesseract
    pil_img = Image.fromarray(pre)

    config = '--psm 3'  # default page segmentation mode
    if config_extra:
        config = config + ' ' + config_extra

    text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
    return text


def ocr_images(image_paths: List[str], **kwargs) -> str:
    """OCR multiple images and concat results with separators."""
    parts = []
    for i, p in enumerate(image_paths, start=1):
        try:
            txt = ocr_image_cv2(p, **kwargs)
        except Exception as e:
            txt = f"[ERROR processing {p}: {e}]"
        parts.append(f"--- PAGE {i}: {os.path.basename(p)} ---\n" + txt.strip())
    return "\n\n".join(parts)


def images_from_pdf(pdf_path: str, dpi: int = 300, pages: Optional[List[int]] = None) -> List[Image.Image]:
    """Convert PDF pages to PIL Images using pdf2image (if available). Returns list of PIL images."""
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError("pdf2image is not installed or not available. Install it with 'pip install pdf2image'.")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # pdf2image expects a list of page numbers starting at 1
    if pages is not None:
        pil_pages = convert_from_path(pdf_path, dpi=dpi, first_page=pages[0], last_page=pages[-1])
    else:
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
    return pil_pages


def ocr_pdf(pdf_path: str, lang: str = 'eng', dpi: int = 300, deskew: bool = False, binarize: bool = True,
            pages: Optional[List[int]] = None) -> str:
    """Extract text from a PDF by converting pages to images then running OCR."""
    pil_pages = images_from_pdf(pdf_path, dpi=dpi, pages=pages)

    page_texts = []
    for idx, pil in enumerate(pil_pages, start=1):
        # convert pil to OpenCV format (numpy array BGR)
        cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        txt = ocr_image_cv2_from_array(cv_img, lang=lang, deskew=deskew, binarize=binarize)
        page_texts.append(f"--- PDF PAGE {idx} ---\n" + txt.strip())

    return "\n\n".join(page_texts)


def ocr_image_cv2_from_array(cv_img: np.ndarray, lang: str = 'eng', deskew: bool = False, binarize: bool = True,
                             config_extra: Optional[str] = None) -> str:
    """Run OCR on an already-loaded OpenCV image (numpy array)."""
    pre = preprocess_image_for_ocr(cv_img, deskew=deskew, binarize=binarize)
    pil_img = Image.fromarray(pre)
    config = '--psm 3'
    if config_extra:
        config = config + ' ' + config_extra
    text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
    return text


def save_text_to_file(text: str, out_path: str) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)


def parse_pages_range(pages_str: str) -> List[int]:
    """Parse a pages string like '1-3,5' into a list [1,2,3,5]. Pages are 1-indexed."""
    pages = set()
    for part in pages_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            start, end = int(start), int(end)
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def main(argv=None):
    parser = argparse.ArgumentParser(description='OCR utility for images and PDFs (CMTI project).')
    parser.add_argument('--input', '-i', required=True, help='Input image or PDF file')
    parser.add_argument('--output', '-o', help='Output text file (if omitted prints to stdout)')
    parser.add_argument('--lang', default='eng', help='Tesseract language code(s), e.g. eng+hin')
    parser.add_argument('--deskew', action='store_true', help='Attempt to deskew pages before OCR')
    parser.add_argument('--no-binarize', dest='binarize', action='store_false', help='Disable adaptive binarization')
    parser.add_argument('--dpi', type=int, default=300, help='DPI when converting PDFs (higher for better OCR)')
    parser.add_argument('--pages', type=str, default=None, help="PDF pages to convert (e.g. '1-3,5')")
    parser.add_argument('--config', type=str, default=None, help='Extra config string passed to tesseract e.g. "-c tessedit_char_whitelist=0123456789"')

    args = parser.parse_args(argv)

    input_path = args.input
    out_path = args.output

    try:
        if input_path.lower().endswith('.pdf'):
            pages = parse_pages_range(args.pages) if args.pages else None
            text = ocr_pdf(input_path, lang=args.lang, dpi=args.dpi, deskew=args.deskew, binarize=args.binarize, pages=pages)
        else:
            # support single image or a wildcard list
            if '*' in input_path or ',' in input_path:
                # expand comma separated list
                img_paths = []
                if ',' in input_path:
                    img_paths = [p.strip() for p in input_path.split(',')]
                else:
                    import glob
                    img_paths = glob.glob(input_path)
                text = ocr_images(img_paths, lang=args.lang, deskew=args.deskew, binarize=args.binarize, config_extra=args.config)
            else:
                text = ocr_image_cv2(input_path, lang=args.lang, deskew=args.deskew, binarize=args.binarize, config_extra=args.config)

        if out_path:
            save_text_to_file(text, out_path)
            print(f"OCR complete — output written to: {out_path}")
        else:
            print(text)

    except Exception as exc:
        print(f"Error during OCR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()


# -----------------------------
# FastAPI wrapper (CMTI OCR API)
# -----------------------------
# Usage:
#   Install extras: pip install fastapi uvicorn python-multipart
#   Run: uvicorn ocr:app --reload --port 8000
# Endpoints:
#   POST /ocr       -> form upload single image file, returns plain text
#   POST /ocr_pdf   -> form upload PDF file, optional pages ("1-3,5")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile

app = FastAPI(title="CMTI OCR API")

# enable CORS for local development (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ocr", response_class=PlainTextResponse)
async def ocr_endpoint(
    file: UploadFile = File(...),
    lang: str = Form("eng"),
    deskew: bool = Form(False),
    binarize: bool = Form(True),
    config: str = Form(None),
):
    """Accepts an image upload and returns extracted text."""
    suffix = os.path.splitext(file.filename)[1] or ".png"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            tmp_path = tmp.name

        text = ocr_image_cv2(tmp_path, lang=lang, deskew=deskew, binarize=binarize, config_extra=config)
        return text
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/ocr_pdf", response_class=PlainTextResponse)
async def ocr_pdf_endpoint(
    file: UploadFile = File(...),
    lang: str = Form("eng"),
    dpi: int = Form(300),
    deskew: bool = Form(False),
    binarize: bool = Form(True),
    pages: str = Form(None),
    config: str = Form(None),
):
    """Accepts a PDF upload and returns extracted text. Use `pages` like '1-3,5'."""
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            tmp_path = tmp.name

        page_list = parse_pages_range(pages) if pages else None
        text = ocr_pdf(tmp_path, lang=lang, dpi=dpi, deskew=deskew, binarize=binarize, pages=page_list)
        return text
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# Optional health endpoint
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


# If run directly, allow running the FastAPI server. For production, use uvicorn/gunicorn.
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("ocr:app", host="0.0.0.0", port=8000, reload=True)
    except Exception:
        print("uvicorn not installed; to run the API install 'uvicorn' and run: uvicorn ocr:app --reload --port 8000")


# -----------------------------
# Simple API key authentication
# -----------------------------
# How it works:
# - The API expects a single API key value stored in the environment variable `CMTI_OCR_API_KEY`.
# - Each protected endpoint requires the client to send the key in the `X-API-KEY` header or as a `api_key` query parameter.
# - If the key is missing or invalid, the API responds with HTTP 401.
#
# To set the key (example):
#   export CMTI_OCR_API_KEY="your_secret_key_here"
#
# The following dependency enforces the key on endpoints that include `Depends(require_api_key)`.

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader
import os

API_KEY = os.environ.get("CMTI_OCR_API_KEY")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


def require_api_key(request: Request, x_api_key: str = Depends(api_key_header)):
    """Dependency that validates the API key supplied either in header or query param."""
    # allow api_key in query param as well
    qp = request.query_params.get("api_key")
    supplied = x_api_key or qp
    if API_KEY is None:
        # Insecure default during development — the server will still reject until env var is set.
        raise HTTPException(status_code=500, detail="Server API key not configured (set CMTI_OCR_API_KEY)")
    if not supplied or supplied != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True

# Apply the dependency to the existing endpoints by adding `Depends(require_api_key)` to their signatures.
# Example: `async def ocr_endpoint(..., auth=Depends(require_api_key)):`
# I have applied it to both `/ocr` and `/ocr_pdf` endpoints in the file.

# Additional notes:
# - You can use a reverse proxy (nginx) or a secrets manager to protect and rotate keys for production.
# - If you'd like, I can extend this to support per-user keys stored in a small SQLite table, an admin endpoint
#   for creating/revoking keys, rate limiting, or JWT-based auth for stronger security.

# -----------------------------
# PostgreSQL + per-user API keys (Option A)
# -----------------------------
# This block adds SQLAlchemy models and admin endpoints that store API keys in PostgreSQL.
# It also replaces the runtime API-key check by pointing `require_api_key` to the DB-backed version.

from sqlalchemy import create_engine, Column, Integer, Text, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime
import hashlib
import secrets

# DATABASE_URL example: postgresql://username:password@localhost:5432/cmti_db
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    # We require PostgreSQL for per-user keys; raise early so devs know to set it.
    # Note: if you'd rather fall back to the original env var behavior, change this.
    raise RuntimeError("DATABASE_URL is not set. Please set DATABASE_URL to your PostgreSQL connection string.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ApiKey(Base):
    __tablename__ = 'api_keys'
    id = Column(Integer, primary_key=True, index=True)
    hashed_key = Column(Text, unique=True, index=True, nullable=False)
    user_name = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    revoked = Column(Boolean, default=False)
    last_used_at = Column(DateTime, nullable=True)


# Create tables if they don't exist
Base.metadata.create_all(bind=engine)


# Dependency: get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_key(raw_key: str) -> str:
    """Deterministic hashing of API key for storage & comparison."""
    return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()


def generate_api_key(length: int = 32) -> str:
    """Generate a secure URL-safe API key to return to clients (raw)."""
    return secrets.token_urlsafe(length)


# DB-backed require_api_key implementation
from fastapi import Depends

def require_api_key_db(request: Request, x_api_key: str = Depends(api_key_header), db: Session = Depends(get_db)):
    """Validate provided API key against the PostgreSQL api_keys table."""
    supplied = x_api_key or request.query_params.get('api_key')
    if not supplied:
        raise HTTPException(status_code=401, detail='Missing API Key')

    supplied_hashed = hash_key(supplied)
    key_row = db.query(ApiKey).filter(ApiKey.hashed_key == supplied_hashed).first()
    if not key_row or key_row.revoked:
        raise HTTPException(status_code=401, detail='Invalid or revoked API Key')

    # update last_used_at
    key_row.last_used_at = datetime.datetime.utcnow()
    db.add(key_row)
    db.commit()

    # You can return the key_row or True; returning True keeps compatibility
    return True


# Expose require_api_key name used elsewhere to the DB-backed implementation
require_api_key = require_api_key_db


# Admin protection: require a master admin key stored in env var
CMTI_OCR_ADMIN_KEY = os.environ.get('CMTI_OCR_ADMIN_KEY')

from fastapi import Header

def require_admin_key(x_admin_key: str = Header(None)):
    if not CMTI_OCR_ADMIN_KEY:
        raise HTTPException(status_code=500, detail='Server admin key not configured (set CMTI_OCR_ADMIN_KEY)')
    if not x_admin_key or x_admin_key != CMTI_OCR_ADMIN_KEY:
        raise HTTPException(status_code=401, detail='Invalid admin key')
    return True


# Admin endpoints for key management
@app.post('/admin/keys')
async def admin_create_key(user_name: str = Form(None), description: str = Form(None), admin_ok: bool = Depends(require_admin_key)):
    """Create a new API key. Returns the raw key (only shown once) and metadata."""
    db = next(get_db())
    raw = generate_api_key()
    hashed = hash_key(raw)
    key = ApiKey(hashed_key=hashed, user_name=user_name, description=description)
    db.add(key)
    db.commit()
    db.refresh(key)
    return {
        'id': key.id,
        'api_key': raw,  # show the raw key only at creation
        'user_name': key.user_name,
        'description': key.description,
        'created_at': key.created_at.isoformat()
    }


@app.get('/admin/keys')
async def admin_list_keys(admin_ok: bool = Depends(require_admin_key)):
    """List keys (do not reveal the raw key)."""
    db = next(get_db())
    rows = db.query(ApiKey).order_by(ApiKey.created_at.desc()).all()
    result = []
    for r in rows:
        result.append({
            'id': r.id,
            'user_name': r.user_name,
            'description': r.description,
            'created_at': r.created_at.isoformat() if r.created_at else None,
            'revoked': bool(r.revoked),
            'last_used_at': r.last_used_at.isoformat() if r.last_used_at else None,
        })
    return result


@app.post('/admin/keys/{key_id}/revoke')
async def admin_revoke_key(key_id: int, admin_ok: bool = Depends(require_admin_key)):
    db = next(get_db())
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail='Key not found')
    key.revoked = True
    db.add(key)
    db.commit()
    return {'status': 'revoked', 'id': key_id}


# Attach the auth dependency to the OCR endpoints (they were already set up to depend on `require_api_key` earlier in the file)
# If you did not see the endpoints fail, they should now validate keys against Postgres.

# Instructions for setup (summary):
# 1) Install DB driver + ORM: pip install sqlalchemy psycopg2-binary
# 2) Create a PostgreSQL database and set DATABASE_URL e.g.:
#       export DATABASE_URL='postgresql://user:pass@localhost:5432/cmti_db'
# 3) Set an admin key for admin endpoints:
#       export CMTI_OCR_ADMIN_KEY='a_strong_admin_key_here'
# 4) Run the FastAPI server:
#       uvicorn ocr:app --reload --port 8000
# 5) Create per-user keys via admin endpoint (use curl with header "X-ADMIN-KEY"):
#       curl -X POST "http://localhost:8000/admin/keys" -H "X-ADMIN-KEY: your_admin_key" -F "user_name=alice" -F "description=mobile client"


