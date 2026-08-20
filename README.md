# PDF Engine Toolbox

PDF Engine Toolbox is the PyMuPDF-based document-processing network service
used by TaxScout. It is free software licensed under
[GNU AGPL-3.0-or-later](LICENSE).
Third-party attribution is recorded in [NOTICE.md](NOTICE.md).

## Corresponding Source offer

Every network response includes a `Link` header with `rel="source"` and an
`X-Source-Code` header. `GET /source` provides the license, public repository,
the exact source commit embedded in the running image, and a downloadable
source archive. `GET /health` exposes the same build commit and source URL.
The source response also includes hash-pinned source archives for PyMuPDF and
its MuPDF engine, as recorded in `third-party-sources.json`.

The complete Corresponding Source is available at no charge from:

https://github.com/TaxScout-ai/pdf-engine-toolbox

For a production build, follow the exact `source_code_url` returned by the
running service. It points to the source revision used for that image. The
repository contains the application source, dependency manifests, tests,
Dockerfile, Compose configuration, and model-download tooling needed to build
and run the service. Secrets and customer data are not part of the source.

## Build and run

Production images must be built only after the source commit is available in
the public repository:

```bash
git clone https://github.com/TaxScout-ai/pdf-engine-toolbox.git
cd pdf-engine-toolbox
git checkout <full-public-commit-sha>
docker build \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t pdf-engine-toolbox:"$(git rev-parse --short HEAD)" .
```

For an immutable Docker Compose run, export the same full public revision first:

```bash
export SOURCE_COMMIT="$(git rev-parse HEAD)"
docker compose up --build
```

The build rejects missing, abbreviated, uppercase, otherwise malformed, or
non-public source revisions. It verifies that the exact revision is reachable
from the public repository before producing the image. The Compose service runs
the verified image without bind-mounting local source or enabling hot reload, so
its reported build commit remains true for the code serving requests.

The third-party source URLs are part of this offer. If an upstream source
archive becomes unavailable, an operator must mirror the hash-identical
archive on a no-charge network server and update the manifest before continuing
to operate that build.

After every deployment, verify the running source offer against the expected
public revision:

```bash
python scripts/verify_agpl_offer.py \
  https://pdf-engine.example.com \
  <full-public-commit-sha>
```

Periodically add `--verify-third-party-hashes` to download and verify the
PyMuPDF and MuPDF source archives. This is intentionally not the default
because the two archives are approximately 150 MB combined.

## Development

Install Python 3.12 dependencies from `requirements.txt`, configure the values
documented in `.env.example`, and run:

```bash
uvicorn app.main:app --reload
```

Development processes outside the container report `development` as their
build identity and link to the repository's default branch.

## License

Copyright (C) 2026 TaxScout.AI Inc. and contributors.

This program is distributed under AGPL-3.0-or-later, without any warranty. See
[LICENSE](LICENSE). If you modify this program and make the modified version
available for users to interact with over a network, AGPL section 13 requires
you to offer those users the Corresponding Source of that modified version.
