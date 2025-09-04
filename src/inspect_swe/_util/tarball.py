# type: ignore

import gzip
import tarfile
from io import BytesIO


def extract_tarball(tarball_bytes: bytes) -> bytes:
    """Extract the binary from a tar.gz archive."""
    # Open the gzip-compressed tarball
    with BytesIO(tarball_bytes) as tarball_io:
        with gzip.open(tarball_io, "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r") as tar:
                # List all members (should be just one file)
                members = tar.getmembers()
                if len(members) != 1:
                    raise ValueError(
                        f"Expected 1 file in tarball, found {len(members)}"
                    )

                # Extract the binary file
                member = members[0]
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Could not extract {member.name}")

                result: bytes = extracted.read()
                return result
