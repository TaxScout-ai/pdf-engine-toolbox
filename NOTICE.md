# Open-source notices

PDF Engine Toolbox is Copyright (C) 2026 TaxScout.AI Inc. and contributors.
It is licensed under GNU AGPL-3.0-or-later; see `LICENSE`.

This service incorporates PyMuPDF and MuPDF. PyMuPDF 1.27.1 is recorded
conservatively as AGPL-3.0-only because its release metadata names GNU Affero
GPL 3.0 without an "or later" grant. MuPDF 1.27.1 expressly grants
AGPL-3.0-or-later. Both remain Copyright their respective contributors and
rights holders. TaxScout does not remove or replace their copyright notices.
The combined running service is therefore identified as AGPL-3.0-only, while
TaxScout's own source remains available under AGPL-3.0-or-later.
Exact versions, source archive locations, and SHA-256 hashes are recorded in
`third-party-sources.json` and returned by the running service at `GET /source`.

No trademark rights are granted by the open-source licenses. Product and
company names belong to their respective owners.


The optional Docling profile includes Docling (MIT, The Docling Contributors),
Heron layout weights (Apache-2.0) and the Docling model repository (which lists
CDLA-Permissive-2.0 and Apache-2.0). Upstream notices and license terms continue
to apply; the model repositories are:

- https://github.com/docling-project/docling
- https://huggingface.co/docling-project/docling-layout-heron
- https://huggingface.co/docling-project/docling-layout-heron-onnx
- https://huggingface.co/docling-project/docling-models

Model cards are preserved alongside the downloaded artifacts. Applicable
license texts are included in `app/services/model_licenses/`.

Pinned artifact revisions are recorded in `scripts/download_docling_models.py`.
Optional package versions are recorded in `requirements-docling.txt`.
