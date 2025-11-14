import ssl, aiohttp, asyncio

async def test():
    connector = aiohttp.TCPConnector(ssl=False)  # 完全关闭验证
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get("https://api.msedgeservices.com") as resp:
            print("Status:", resp.status)

asyncio.run(test())
