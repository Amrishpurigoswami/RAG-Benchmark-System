import asyncio
import cognee

async def main():

    print("=" * 60)
    print("GRAPH STATUS")
    print("=" * 60)

    status = await cognee.datasets.get_status("main_dataset")

    print(status)

asyncio.run(main())