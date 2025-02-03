import asyncio
import copy
import json
import time
import traceback
import typing

import bittensor as bt
import pydantic
import requests
import yaml
from bittensor_wallet import Keypair

from generation_models.utils import base64_to_pil_image

MODEL_CONFIG = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)


class Information(bt.Synapse):
    request_dict: dict = {}
    response_dict: dict = {}


class ImageGenerating(bt.Synapse):
    prompt: str = pydantic.Field(
        default="",
        title="Prompt",
        description="Prompt for generation",
    )
    seed: int = pydantic.Field(
        default=0,
        title="Seed",
        description="Seed for generation",
    )
    model_name: str = pydantic.Field(
        default="",
        title="",
        description="Name of the model used for generation",
    )
    conditional_image: str = pydantic.Field(
        default="",
        title="Base64 Image",
        description="Base64 encoded image",
    )
    pipeline_type: str = pydantic.Field(
        default="txt2img",
        title="Pipeline Type",
        description="Type of pipeline used for generation, eg: txt2img, img2img, controlnet_txt2img",
    )
    pipeline_params: dict = pydantic.Field(
        default={},
        title="Pipeline Params",
        description="Dictionary of additional parameters for diffusers pipeline",
    )
    request_dict: dict = pydantic.Field(
        default={},
        title="Dictionary contains request",
        description="Dict contains arbitary information",
    )
    response_dict: dict = pydantic.Field(
        default={},
        title="Dictionary contains response",
        description="Dict contains arbitary information",
    )
    image: str = pydantic.Field(
        default="",
        title="Base64 Image",
        description="Base64 encoded image",
    )

    def miner_update(self, update: dict):
        # return self.copy(update=update)
        return self.model_copy(update=update)

    def deserialize_input(self) -> dict:
        return self.deserialize()

    def limit_params(self):
        for k, v in self.pipeline_params.items():
            if k == "num_inference_steps":
                self.pipeline_params[k] = min(50, v)
            if k == "width":
                self.pipeline_params[k] = min(1536, v)
            if k == "height":
                self.pipeline_params[k] = min(1536, v)
        self.pipeline_params = self.pipeline_params

    def deserialize(self) -> dict:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "model_name": self.model_name,
            "pipeline_type": self.pipeline_type,
            "pipeline_params": self.pipeline_params,
            "conditional_image": self.conditional_image,
            "image": self.image,
            "response_dict": self.response_dict,
        }

    def deserialize_response(self):
        return {
            "image": self.image,
            "response_dict": self.response_dict,
        }

    def store_response(self, storage_url: str, uid, validator_uid, keypair: Keypair):
        if self.model_name == "GoJourney":
            storage_url = storage_url + "/upload-go-journey-item"
            data = {
                "metadata": {
                    "miner_uid": uid,
                    "validator_uid": validator_uid,
                    "prompt": self.prompt,
                    "seed": self.seed,
                    "model_name": self.model_name,
                },
                "output": self.response_dict
            }
        else:
            if self.model_name != "FluxSchnell":
                return
            storage_url = storage_url + "/upload-base64-item"
            data = {
                "image": self.image,
                "metadata": {
                    "miner_uid": uid,
                    "validator_uid": validator_uid,
                    "model_name": self.model_name,
                    "prompt": self.prompt,
                    "seed": self.seed,
                    "pipeline_type": self.pipeline_type,
                    "pipeline_params": self.pipeline_params,
                }
            }
        serialized_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
        nonce = str(time.time_ns())
        # Calculate validator 's signature
        message = f"{serialized_data}{keypair.ss58_address}{nonce}"
        signature = f"0x{keypair.sign(message).hex()}"
        # Add validator 's signature
        data["nonce"] = nonce
        data["signature"] = signature
        try:
            response = requests.post(storage_url, json=data)
            response.raise_for_status()
        except Exception as e:
            print(f"Error in storing response: {e}")
            traceback.print_exc()


class TextGenerating(bt.Synapse):
    # Required request input, filled by sending dendrite caller.
    prompt_input: str = ""
    # Optional request output, filled by recieving axon.
    seed: int = 0
    request_dict: dict = {}
    model_name: str = ""
    prompt_output: typing.Optional[dict] = {}
    pipeline_params: dict = {}

    def miner_update(self, update: dict):
        self.prompt_output = update

    def deserialize_input(self) -> dict:
        deserialized_input = {
            "model": MODEL_CONFIG[self.model_name].get("repo_id", self.model_name),
            "prompt": [
                self.prompt_input,
            ],
        }
        deserialized_input.update(self.pipeline_params)
        return deserialized_input
    
    def limit_params(self):
        for k, v in self.pipeline_params.items():
            if k == "max_tokens":
                self.pipeline_params[k] = min(4096, v)
        self.pipeline_params = self.pipeline_params
        
    def deserialize(self) -> dict:
        """
        Deserialize the prompt output. This method retrieves the response from
        the miner in the form of prompt_output, deserializes it and returns it
        as the output of the dendrite.query() call.
        Returns:
        - dict: The deserialized response, which in this case is the value of prompt_output.
        """

        return {
            "prompt_output": self.prompt_output,
            "prompt_input": self.prompt_input,
            "model_name": self.model_name,
        }

    def deserialize_response(self):
        minimized_prompt_output: dict = copy.deepcopy(self.prompt_output)
        minimized_prompt_output['choices'][0].pop("logprobs")
        return {
            "prompt_output": minimized_prompt_output,
            "prompt_input": self.prompt_input,
            "model_name": self.model_name,
        }

    def store_response(self, storage_url: str, uid, validator_uid, keypair: Keypair):
        pass

class MultiModalGenerating(bt.Synapse):
    prompt: str = pydantic.Field(
        default="",
        title="Prompt",
        description="Prompt for generation",
    )
    image_url: str = pydantic.Field(
        default="",
        title="",
        description="URL of conditional image",
    )

    seed: int = pydantic.Field(
        default=0,
        title="Seed",
        description="Seed for generation",
    )

    request_dict: dict = pydantic.Field(
        default={},
        title="Dictionary contains request",
        description="Dict contains arbitary information",
    )

    model_name: str = pydantic.Field(
        default="",
        title="",
        description="Name of the model used for generation",
    )

    pipeline_params: dict = pydantic.Field(
        default={},
        title="Pipeline Params",
        description="Dictionary of additional parameters for generation",
    )
    pipeline_type: str = pydantic.Field(
        default="visual_question_answering",
        title="Pipeline Type",
        description="Type of pipeline used for generation",
    )

    prompt_output: typing.Optional[dict] = {}

    def miner_update(self, update: dict):
        self.prompt_output = update

    def deserialize_input(self) -> dict:
        message_content = [
            {
                "type": "text",
                "text": self.prompt
            }
        ]
        if self.image_url:
            message_content.append({
            "type": "image_url",
            "image_url": {
                "url": self.image_url
            },
        })

        messages = [
            {
                "role": "user", 
                "content": message_content
            }
        ]

        deserialized_input = {
            "model": MODEL_CONFIG[self.model_name].get("repo_id", self.model_name),
            "prompt": [
                self.prompt,
            ],
            "image_url": self.image_url,
            "pipeline_type": self.pipeline_type,
            "seed": self.seed,
            "messages": messages
        }
        logprobs = self.pipeline_params.get("logprobs")
        if logprobs:
            self.pipeline_params["top_logprobs"] = copy.deepcopy(self.pipeline_params["logprobs"])
            self.pipeline_params["logprobs"] = True        
        deserialized_input.update(self.pipeline_params)

        return deserialized_input
    
    def limit_params(self):
        for k, v in self.pipeline_params.items():
            if k == "max_tokens":
                self.pipeline_params[k] = min(8192, v)
        self.pipeline_params = self.pipeline_params
        
    def deserialize(self) -> dict:
        return {
            "prompt_output": self.prompt_output,
            "prompt": self.prompt,
            "model_name": self.model_name,
            "seed": self.seed
        }

    def deserialize_response(self):
        minimized_prompt_output: dict = copy.deepcopy(self.prompt_output)
        minimized_prompt_output['choices'][0].pop("logprobs")
        return {
            "prompt_output": minimized_prompt_output,
            "prompt_input": self.prompt,
            "model_name": self.model_name,
        }

    def store_response(self, storage_url: str, uid, validator_uid, keypair: Keypair):
        storage_url = storage_url + "/upload-multimodal-item"
        minimized_prompt_output: dict = copy.deepcopy(self.prompt_output)
        minimized_prompt_output['choices'][0].pop("logprobs")
        data = {
            "prompt_input": self.prompt,
            "image_url": self.image_url,
            "prompt_output": minimized_prompt_output,
            "metadata": {
                "miner_uid": uid,
                "validator_uid": validator_uid,
                "model": MODEL_CONFIG[self.model_name].get("repo_id", self.model_name),
                "model_name": self.model_name,
                "pipeline_params": self.pipeline_params,
            }
        }
        serialized_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
        nonce = str(time.time_ns())
        # Calculate validator 's signature
        message = f"{serialized_data}{keypair.ss58_address}{nonce}"
        signature = f"0x{keypair.sign(message).hex()}"
        # Add validator 's signature
        data["nonce"] = nonce
        data["signature"] = signature
        try:
            response = requests.post(storage_url, json=data)
            response.raise_for_status()
        except Exception as e:
            print(f"Error in storing response: {e}")
            traceback.print_exc()

class my_dendrite(bt.dendrite):
    def query_axons_batch(self, axons, synapses, deserialize=True, timeout=12):
        """
        Makes a synchronous request to multiple target Axons and returns the server responses.

        Cleanup is automatically handled and sessions are closed upon completed requests.

        Args:
            axons (Union[list[Union[bittensor.core.chain_data.axon_info.AxonInfo, 'bittensor.core.axon.Axon']], Union['bittensor.core.chain_data.axon_info.AxonInfo', 'bittensor.core.axon.Axon']]): The list of target Axon information.
            synapse (Optional[bittensor.core.synapse.Synapse]): The Synapse object. Defaults to :func:`Synapse()`.
            timeout (Optional[float]): The request timeout duration in seconds. Defaults to ``12.0`` seconds.

        Returns:
            Union[bittensor.core.synapse.Synapse, list[bittensor.core.synapse.Synapse]]: If a single target axon is provided, returns the response from that axon. If multiple target axons are provided, returns a list of responses from all target axons.
        """
        result = None
        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.forward(axons, synapses, deserialize, timeout))
        except Exception:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = new_loop.run_until_complete(self.forward(axons, synapses, deserialize, timeout))
            new_loop.close()
        finally:
            self.close_session()
            return result  # type: ignore
        
    async def forward(
        self,
        axons,
        synapses,
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False
    ):
        """
        Asynchronously sends requests to one or multiple Axons and collates their responses.

        This function acts as a bridge for sending multiple requests concurrently or sequentially
        based on the provided parameters. It checks the type of the target Axons, preprocesses
        the requests, and then sends them off. After getting the responses, it processes and
        collates them into a unified format.

        When querying an Axon that sends a single response, this function returns a Synapse object
        containing the response data. If multiple Axons are queried, a list of Synapse objects is
        returned, each containing the response from the corresponding Axon.

        For example::

            ...
            import bittensor
            wallet = bittensor.Wallet()                     # Initialize a wallet
            synapse = bittensor.Synapse(...)                # Create a synapse object that contains query data
            dendrite = bittensor.Dendrite(wallet = wallet)  # Initialize a dendrite instance
            netuid = ...                                    # Provide subnet ID
            metagraph = bittensor.Metagraph(netuid)         # Initialize a metagraph instance
            axons = metagraph.axons                         # Create a list of axons to query
            responses = await dendrite(axons, synapse)      # Send the query to all axons and await the responses

        When querying an Axon that sends back data in chunks using the Dendrite, this function
        returns an AsyncGenerator that yields each chunk as it is received. The generator can be
        iterated over to process each chunk individually.

        For example::

            ...
            dendrite = bittensor.Dendrite(wallet = wallet)
            async for chunk in dendrite.forward(axons, synapse, timeout, deserialize, run_async, streaming):
                # Process each chunk here
                print(chunk)

        Args:
            axons (Union[list[Union[bittensor.core.chain_data.axon_info.AxonInfo, bittensor.core.axon.Axon]], Union[bittensor.core.chain_data.axon_info.AxonInfo, bittensor.core.axon.Axon]]): The target Axons to send requests to. Can be a single Axon or a list of Axons.
            synapse (bittensor.core.synapse.Synapse): The Synapse object encapsulating the data. Defaults to a new :func:`Synapse` instance.
            timeout (float): Maximum duration to wait for a response from an Axon in seconds. Defaults to ``12.0``.
            deserialize (bool): Determines if the received response should be deserialized. Defaults to ``True``.
            run_async (bool): If ``True``, sends requests concurrently. Otherwise, sends requests sequentially. Defaults to ``True``.
            streaming (bool): Indicates if the response is expected to be in streaming format. Defaults to ``False``.

        Returns:
            Union[AsyncGenerator, bittensor.core.synapse.Synapse, list[bittensor.core.synapse.Synapse]]: If a single `Axon` is targeted, returns its response.
            If multiple Axons are targeted, returns a list of their responses.
        """
        is_list = True
        # If a single axon is provided, wrap it in a list for uniform processing
        if not isinstance(axons, list):
            is_list = False
            axons = [axons]

        # Get responses for all axons.
        responses = await asyncio.gather(
                *(self.call(target_axon, synapse, timeout, deserialize) for (target_axon, synapse) in zip(axons, synapses))
            ) 
        # Return the single response if only one axon was targeted, else return all responses
        return responses[0] if len(responses) == 1 and not is_list else responses  # type: ignore
