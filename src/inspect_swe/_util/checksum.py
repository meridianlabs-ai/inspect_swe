import hashlib


class ChecksumMismatchError(ValueError):
    """Downloaded (or shared-resolution) bytes did not match the expected digest.

    Distinct from other ``ValueError``s raised in this module so callers can
    single out an integrity failure (tampered/corrupted download) from
    ordinary network or resolution failures — an integrity failure must
    never be papered over by falling back to unverified cached bytes.
    """


def verify_checksum(data: bytes, expected_checksum: str) -> bool:
    actual_checksum = hashlib.sha256(data).hexdigest()
    return actual_checksum == expected_checksum
