# AsyncIO in Python
import time
import asyncio
# in order to run aysnc functions we need to run an event loop
# manages and runs for events (functions)
def sync_function(param):
    print("This is sync function")
    time.sleep(0.1)
    return f"Sync res: {param}"


async def async_function(param):
    print("This is async function")
    await asyncio.sleep(0.1)
    return f"Async res: {param}"

async def main():
    #sync_res = sync_function("Test")
    #print(sync_res)
    # __await__() : an object has to be awaitable.
    loop = asyncio.get_running_loop()
    future = loop.create_future() 
    print(f"Empty future: {future}")
    future.set_result = await future # pause exection of curr function, give control to control loop until another tasks ends
    print(future)


    

    



if __name__ == "__main__":
    asyncio.run(main()) # starting event loop: control returns to event loop which finds another task to start or resume
    # we need concurrency : so we use await keyword