"""PDF/A-2b output through Ghostscript (TAX-5637).

PyMuPDF cannot write a conforming PDF/A, so this shells out to Ghostscript's
pdfwrite device: fonts embedded, colours converted to RGB, an sRGB output
intent and the PDF/A identification in the XMP metadata. Encrypted input is
refused, since PDF/A forbids encryption.
"""

import glob
import os
import shutil
import subprocess
import tempfile

import fitz

GS_TIMEOUT_SECONDS = 180

_ICC_CANDIDATES = [
    "/usr/share/color/icc/ghostscript/srgb.icc",
    "/usr/share/ghostscript/*/iccprofiles/srgb.icc",
    "/usr/local/share/ghostscript/*/iccprofiles/srgb.icc",
]


class PdfaConversionError(Exception):
    """The document could not be converted to PDF/A."""


def _srgb_profile() -> str:
    for pattern in _ICC_CANDIDATES:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                return path
    raise PdfaConversionError("Ghostscript sRGB profile not found")


def _definition(icc_path: str) -> str:
    # The standard PDFA_def.ps with the profile path filled in.
    escaped = icc_path.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"""%!
/ICCProfile ({escaped}) def
[/_objdef {{icc_PDFA}} /type /stream /OBJ pdfmark
[{{icc_PDFA}} << /N 3 >> /PUT pdfmark
[{{icc_PDFA}} ICCProfile (r) file /PUT pdfmark
[/_objdef {{OutputIntent_PDFA}} /type /dict /OBJ pdfmark
[{{OutputIntent_PDFA}} <<
  /Type /OutputIntent
  /S /GTS_PDFA1
  /DestOutputProfile {{icc_PDFA}}
  /OutputConditionIdentifier (sRGB)
>> /PUT pdfmark
[{{Catalog}} << /OutputIntents [ {{OutputIntent_PDFA}} ] >> /PUT pdfmark
"""


def pdf_to_pdfa(pdf_bytes: bytes) -> bytes:
    """Return the document as PDF/A-2b."""
    gs = shutil.which("gs")
    if gs is None:
        raise PdfaConversionError("Ghostscript is not installed")
    try:
        probe = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise PdfaConversionError(f"Cannot open PDF: {e}") from e
    if probe.needs_pass or probe.is_encrypted:
        raise PdfaConversionError(
            "A password-protected PDF cannot become PDF/A; remove the password first"
        )
    probe.close()

    icc = _srgb_profile()
    with tempfile.TemporaryDirectory(prefix="pdfa-") as work:
        src = os.path.join(work, "in.pdf")
        dst = os.path.join(work, "out.pdf")
        definition = os.path.join(work, "PDFA_def.ps")
        with open(src, "wb") as f:
            f.write(pdf_bytes)
        with open(definition, "w", encoding="utf-8") as f:
            f.write(_definition(icc))
        cmd = [
            gs,
            "-dPDFA=2",
            "-dBATCH",
            "-dNOPAUSE",
            "-dNOOUTERSAVE",
            "-dQUIET",
            "-dPDFACompatibilityPolicy=1",
            "-sColorConversionStrategy=RGB",
            "-sDEVICE=pdfwrite",
            f"--permit-file-read={icc}",
            f"-sOutputFile={dst}",
            definition,
            src,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=GS_TIMEOUT_SECONDS, check=False
            )
        except subprocess.TimeoutExpired as e:
            raise PdfaConversionError("PDF/A conversion took too long") from e
        if result.returncode != 0 or not os.path.isfile(dst):
            detail = result.stderr.decode("utf-8", "replace")[-300:]
            raise PdfaConversionError(f"Ghostscript failed: {detail}")
        with open(dst, "rb") as f:
            return f.read()
