import httpx


async def download_file(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.content


async def download_text_file(url: str) -> str:
    return (await download_file(url)).decode("utf-8")
