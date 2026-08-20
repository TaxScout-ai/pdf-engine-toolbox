"""Public AGPL Corresponding Source offer."""

from fastapi import APIRouter

from app.build_identity import (
    LICENSE_IDENTIFIER,
    PROJECT_LICENSE_IDENTIFIER,
    REPOSITORY_URL,
    corresponding_license_url,
    corresponding_source_archive_url,
    corresponding_source_url,
    read_build_commit,
    read_third_party_sources,
    third_party_source_manifest_url,
)
from app.models.responses import SourceOfferResponse

router = APIRouter()


@router.get("/source", response_model=SourceOfferResponse)
async def source_offer():
    """Offer the Corresponding Source to every network user at no charge."""
    build_commit = read_build_commit()
    return SourceOfferResponse(
        license=LICENSE_IDENTIFIER,
        project_license=PROJECT_LICENSE_IDENTIFIER,
        license_url=corresponding_license_url(build_commit),
        repository_url=REPOSITORY_URL,
        build_commit=build_commit,
        source_code_url=corresponding_source_url(build_commit),
        source_archive_url=corresponding_source_archive_url(build_commit),
        third_party_source_manifest_url=third_party_source_manifest_url(build_commit),
        third_party_sources=read_third_party_sources(),
    )
