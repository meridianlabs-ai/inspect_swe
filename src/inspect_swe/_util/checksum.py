import hashlib


def verify_checksum(data: bytes, expected_checksum: str) -> bool:
    actual_checksum = hashlib.sha256(data).hexdigest()
    return actual_checksum == expected_checksum
