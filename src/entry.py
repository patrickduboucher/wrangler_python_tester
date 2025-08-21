from workers import WorkerEntrypoint, Response
from hello import hello

class Default(WorkerEntrypoint):
    async def fetch(self, request):
        name = (await request.json().name)
        return Response(hello(name))