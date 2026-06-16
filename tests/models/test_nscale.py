
from pydantic_ai.providers.nscale import NscaleProvider
from pydantic import BaseModel
from pydantic_ai import Agent


if __name__=="__main__":
    class MyModel(BaseModel):
        city: str
        country: str

    agent = Agent("nscale:meta-llama/Llama-3.1-8B-Instruct", output_type=MyModel)

    result = agent.run_sync("Miami of England.")

    print(result.output)