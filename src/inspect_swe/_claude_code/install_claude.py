#!/usr/bin/env python3

import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, cast

# Constants
INSTALL_SCRIPT_URL = "https://claude.ai/install.sh"
CACHE_DIR = Path.home() / ".claude" / "downloads"
# Fallback GCS bucket in case we can't fetch from install.sh
FALLBACK_GCS_BUCKET = "https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases"


def run_docker_exec(container_name: str, command: str) -> str:
    """Execute a command in the Docker container and return output."""
    cmd = ["docker", "exec", container_name, "bash", "-c", command]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def detect_platform(container_name: str) -> str:
    """Detect the platform (OS and architecture) of the container."""
    # Get OS
    os_name = run_docker_exec(container_name, "uname -s")
    if os_name == "Darwin":
        os_type = "darwin"
    elif os_name == "Linux":
        os_type = "linux"
    else:
        raise ValueError(f"Unsupported OS: {os_name}")

    # Get architecture
    arch = run_docker_exec(container_name, "uname -m")
    if arch in ["x86_64", "amd64"]:
        arch_type = "x64"
    elif arch in ["arm64", "aarch64"]:
        arch_type = "arm64"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Check for musl on Linux
    if os_type == "linux":
        # Check for musl libc
        musl_check_cmd = (
            "if [ -f /lib/libc.musl-x86_64.so.1 ] || "
            "[ -f /lib/libc.musl-aarch64.so.1 ] || "
            "ldd /bin/ls 2>&1 | grep -q musl; then "
            "echo 'musl'; else echo 'glibc'; fi"
        )
        libc_type = run_docker_exec(container_name, musl_check_cmd)
        if libc_type == "musl":
            platform = f"linux-{arch_type}-musl"
        else:
            platform = f"linux-{arch_type}"
    else:
        platform = f"{os_type}-{arch_type}"

    return platform


def download_file(url: str) -> bytes:
    """Download a file from the given URL and return its contents."""
    with urllib.request.urlopen(url) as response:
        return cast(bytes, response.read())


def get_gcs_bucket_from_install_script() -> str:
    """Fetch the install.sh script and extract the GCS_BUCKET URL.

    Falls back to hardcoded URL if extraction fails.
    """
    try:
        print("Fetching install script to discover GCS bucket...")
        script_content = download_file(INSTALL_SCRIPT_URL).decode("utf-8")

        # Look for GCS_BUCKET= line in the script
        # Pattern matches: GCS_BUCKET="https://storage.googleapis.com/..."
        pattern = r'GCS_BUCKET="(https://storage\.googleapis\.com/[^"]+)"'
        match = re.search(pattern, script_content)

        if match:
            gcs_bucket = match.group(1)
            print(f"Discovered GCS bucket: {gcs_bucket}")
            return gcs_bucket
        else:
            print("Could not extract GCS bucket from install script, using fallback")
            return FALLBACK_GCS_BUCKET

    except Exception as e:
        print(f"Error fetching install script: {e}, using fallback")
        return FALLBACK_GCS_BUCKET


def validate_target(target: str) -> bool:
    """Validate the target parameter format."""
    pattern = r"^(stable|latest|[0-9]+\.[0-9]+\.[0-9]+(-[^[:space:]]+)?)$"
    return bool(re.match(pattern, target))


def get_version(gcs_bucket: str, target: str = "stable") -> str:
    """Get the actual version to install based on the target."""
    if not validate_target(target):
        raise ValueError(f"Invalid target: {target}")

    # Always download stable version first (it has the most up-to-date installer)
    stable_url = f"{gcs_bucket}/stable"
    stable_version = download_file(stable_url).decode("utf-8").strip()

    if target == "stable" or target == stable_version:
        return stable_version
    elif target == "latest":
        # For latest, we'd need to check the latest version
        # For now, we'll use stable as the implementation
        return stable_version
    else:
        # Specific version requested
        return target


def get_checksum_from_manifest(manifest_json: str, platform: str) -> str:
    """Extract the checksum for the given platform from the manifest."""
    manifest = json.loads(manifest_json)

    if "platforms" not in manifest:
        raise ValueError("Invalid manifest: missing platforms")

    if platform not in manifest["platforms"]:
        raise ValueError(f"Platform {platform} not found in manifest")

    checksum = manifest["platforms"][platform].get("checksum")

    if not checksum or not re.match(r"^[a-f0-9]{64}$", checksum):
        raise ValueError(f"Invalid checksum for platform {platform}")

    return str(checksum)


def verify_checksum(data: bytes, expected_checksum: str) -> bool:
    """Verify the SHA256 checksum of the data."""
    actual_checksum = hashlib.sha256(data).hexdigest()
    return actual_checksum == expected_checksum


def get_cached_binary_path(version: str, platform: str) -> Path:
    """Get the path where a binary would be cached."""
    return CACHE_DIR / f"claude-{version}-{platform}"


def get_cached_binary(
    version: str, platform: str, expected_checksum: str
) -> Optional[bytes]:
    """
    Check if we have a cached binary and verify its checksum.

    Returns the binary data if valid, None otherwise.
    """
    cache_path = get_cached_binary_path(version, platform)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            binary_data = f.read()

        # Verify the cached binary still has the correct checksum
        if verify_checksum(binary_data, expected_checksum):
            # Update access time so this file is considered "recently used"
            cache_path.touch()
            print(f"Using cached binary from {cache_path}")
            return binary_data
        else:
            print("Cached binary checksum mismatch, will re-download")
            cache_path.unlink()  # Remove invalid cache file
            return None
    except Exception as e:
        print(f"Error reading cached binary: {e}")
        return None


def cleanup_old_cache_files(keep_count: int = 3) -> None:
    """
    Remove old cached binaries, keeping only the most recent ones.

    Keeps the specified number of most recently accessed files.
    """
    if not CACHE_DIR.exists():
        return

    # Get all claude binary files in cache
    cache_files = list(CACHE_DIR.glob("claude-*"))

    if len(cache_files) <= keep_count:
        return  # Nothing to clean up

    # Sort by access time (most recently accessed last)
    cache_files.sort(key=lambda f: f.stat().st_atime)

    # Remove oldest files
    files_to_remove = cache_files[:-keep_count]
    for file_path in files_to_remove:
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            file_path.unlink()
            print(f"Removed old cache file: {file_path.name} ({file_size_mb:.1f} MB)")
        except Exception as e:
            print(f"Error removing cache file {file_path}: {e}")


def save_to_cache(binary_data: bytes, version: str, platform: str) -> None:
    """Save a binary to the cache directory and clean up old files."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cached_binary_path(version, platform)

    with open(cache_path, "wb") as f:
        f.write(binary_data)

    print(f"Saved binary to cache: {cache_path}")

    # Clean up old cache files, keeping only the 3 most recent
    cleanup_old_cache_files(keep_count=3)


def transfer_binary(container_name: str, binary_data: bytes, target_path: str) -> None:
    """Transfer binary data to the container."""
    # Use a temporary file and docker cp
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(binary_data)
        tmp_file_path = tmp_file.name

    try:
        # Copy file to container
        subprocess.run(
            ["docker", "cp", tmp_file_path, f"{container_name}:{target_path}"],
            check=True,
        )
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)


def install_claude(container_name: str, binary_path: str) -> None:
    """Install claude binary and verify it works."""
    # Copy binary to /usr/local/bin for system-wide access
    run_docker_exec(container_name, f"cp {binary_path} /usr/local/bin/claude")
    run_docker_exec(container_name, "chmod +x /usr/local/bin/claude")

    # Clean up the temporary binary
    run_docker_exec(container_name, f"rm -f {binary_path}")

    # Verify installation and initialize config
    try:
        # Check version
        version_output = run_docker_exec(container_name, "claude --version")
        print(f"Claude installed successfully: {version_output}")

        # Initialize config files/directories by running config list
        run_docker_exec(container_name, "claude config list")
        print("Claude configuration initialized")

    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not verify claude installation: {e}")
        raise ValueError("Claude installation verification failed") from e


def main(container_name: str, target: str = "stable") -> None:
    """Main function to orchestrate the Claude installation."""
    print(f"Installing Claude Code in container: {container_name}")
    print(f"Target: {target}")

    # Step 0: Get GCS bucket URL
    gcs_bucket = get_gcs_bucket_from_install_script()

    # Step 1: Detect platform
    print("Detecting platform...")
    platform = detect_platform(container_name)
    print(f"Platform: {platform}")

    # Step 2: Get version
    print("Determining version...")
    version = get_version(gcs_bucket, target)
    print(f"Version: {version}")

    # Step 3: Download and parse manifest
    print("Downloading manifest...")
    manifest_url = f"{gcs_bucket}/{version}/manifest.json"
    manifest_json = download_file(manifest_url).decode("utf-8")

    # Step 4: Get checksum for platform
    print("Extracting checksum...")
    expected_checksum = get_checksum_from_manifest(manifest_json, platform)

    # Step 5: Check cache or download binary
    binary_data = get_cached_binary(version, platform, expected_checksum)

    if binary_data is None:
        # Not in cache or invalid, need to download
        print(f"Downloading Claude binary for {platform}...")
        binary_url = f"{gcs_bucket}/{version}/{platform}/claude"
        binary_data = download_file(binary_url)

        # Step 6: Verify checksum
        print("Verifying checksum...")
        if not verify_checksum(binary_data, expected_checksum):
            raise ValueError("Checksum verification failed")
        print("Checksum verified successfully")

        # Save to cache for future use
        save_to_cache(binary_data, version, platform)
    else:
        print("Checksum already verified for cached binary")

    # Step 7: Transfer binary to container
    print("Transferring binary to container...")
    binary_path = f"/tmp/claude-{version}-{platform}"
    transfer_binary(container_name, binary_data, binary_path)

    # Step 8: Install
    print("Installing Claude Code...")
    install_claude(container_name, binary_path)

    print("\n✅ Installation complete!")


if __name__ == "__main__":
    # Test code - replace with your actual container name
    test_container = "inspect-intervention-izedw74-default-1"

    # You can test with different targets
    # main(test_container, "stable")
    # main(test_container, "latest")
    # main(test_container, "1.0.0")

    # Default test
    main(test_container, "stable")
