import asyncio
import json
import math
import pickle
import queue
import random
import threading
import time
import traceback
from datetime import datetime

import bittensor as bt
import numpy as np
import torch
import yaml

import image_generation_subnet as ig_subnet
from generation_models.utils import random_image_size
from image_generation_subnet.base.validator import BaseValidatorNeuron
from image_generation_subnet.validator import MinerManager
from image_generation_subnet.validator.offline_challenge import (
    get_backup_challenge_vqa, get_backup_image, get_backup_llm_prompt,
    get_backup_prompt)
from neurons.validator.validator_proxy import ValidatorProxy
from services.offline_rewarding.redis_client import RedisClient
from services.offline_rewarding.reward_app import RewardApp

MODEL_CONFIGS = yaml.load(
    open("generation_models/configs/model_config.yaml"), yaml.FullLoader
)


class QueryItem:
    def __init__(self, uid: int, should_reward: bool = False):
        self.uid = uid
        self.should_reward = should_reward # currently only used for synthetic query

class QueryQueue:
    def __init__(self, model_names: list[str], time_per_loop: int = 600):
        self.synthentic_queue: dict[str, queue.Queue[QueryItem]] = {
            model_name: queue.Queue() for model_name in model_names
        }
        self.proxy_queue: dict[str, queue.Queue[QueryItem]] = {
            model_name: queue.Queue() for model_name in model_names
        }
        self.synthentic_rewarded = []
        self.time_per_loop = time_per_loop
        self.total_uids_remaining = 0

    def update_queue(self, all_uids_info):
        self.total_uids_remaining = 0
        self.synthentic_rewarded = []
        for q in self.synthentic_queue.values():
            q.queue.clear()
        for q in self.proxy_queue.values():
            q.queue.clear()
        for uid, info in all_uids_info.items():
            if not info["model_name"]:
                continue
            synthentic_model_queue = self.synthentic_queue.setdefault(
                info["model_name"], queue.Queue()
            )
            proxy_model_queue = self.proxy_queue.setdefault(
                info["model_name"], queue.Queue()
            )
            synthetic_rate_limit, proxy_rate_limit = self.get_rate_limit_by_type(
                info["rate_limit"]
            )

            # DEBUG
            synthetic_rate_limit = 5
            for _ in range(int(synthetic_rate_limit)):
                if _ < 5:
                    synthentic_model_queue.put(QueryItem(uid=uid, should_reward=True))
                else:
                    synthentic_model_queue.put(QueryItem(uid=uid, should_reward=False))
                    self.synthentic_rewarded.append(uid)
            # END DEBUG


            # for _ in range(int(synthetic_rate_limit)):
            #     if uid in self.synthentic_rewarded:
            #         synthentic_model_queue.put(QueryItem(uid=uid, should_reward=False))
            #     else:
            #         synthentic_model_queue.put(QueryItem(uid=uid, should_reward=True))
            #         self.synthentic_rewarded.append(uid)

            for _ in range(int(proxy_rate_limit)):
                proxy_model_queue.put(QueryItem(uid=uid))
        # Shuffle the queue
        for model_name, q in self.synthentic_queue.items():
            random.shuffle(q.queue)
            self.total_uids_remaining += len(q.queue)
            bt.logging.info(
                f"- Model {model_name} has {len(q.queue)} uids remaining for synthentic"
            )
        for model_name, q in self.proxy_queue.items():
            random.shuffle(q.queue)
            bt.logging.info(
                f"- Model {model_name} has {len(q.queue)} uids remaining for organic"
            )

    def get_batch_query(self, batch_size: int):
        if not self.total_uids_remaining:
            return
        more_data = True
        while more_data:
            more_data = False
            for model_name, q in self.synthentic_queue.items():
                if q.empty():
                    continue
                time_to_sleep = self.time_per_loop * (
                    min(batch_size / (self.total_uids_remaining + 1), 1)
                )
                uids_to_query = []
                should_rewards = []

                while len(uids_to_query) < batch_size and not q.empty():
                    more_data = True
                    query_item = q.get()
                    uids_to_query.append(query_item.uid)
                    should_rewards.append(query_item.should_reward)

                yield model_name, uids_to_query, should_rewards, time_to_sleep

    def get_query_for_proxy(self, model_name):
        synthentic_q = self.synthentic_queue[model_name]
        proxy_q = self.proxy_queue[model_name]
        while not synthentic_q.empty():
            query_item = synthentic_q.get()
            should_reward = query_item.uid not in self.synthentic_rewarded
            yield query_item.uid, should_reward
        while not proxy_q.empty():
            query_item = proxy_q.get()
            yield query_item.uid, False

    def get_rate_limit_by_type(self, rate_limit):
        synthentic_rate_limit = max(1, int(math.floor(rate_limit * 0.8)) - 1)
        synthentic_rate_limit = max(
            rate_limit - synthentic_rate_limit, synthentic_rate_limit
        )
        proxy_rate_limit = rate_limit - synthentic_rate_limit
        return synthentic_rate_limit, proxy_rate_limit


def initialize_challenge_urls(config):
    challenge_urls = {
        "txt2img": {
            "main": [config.challenge.prompt],
            "backup": [get_backup_prompt],
        },
        "img2img": {
            "main": [config.challenge.prompt, config.challenge.image],
            "backup": [get_backup_prompt, get_backup_image],
        },
        "instantid": {
            "main": [
                config.challenge.prompt,
                config.challenge.image,
            ],
            "backup": [get_backup_prompt, get_backup_image],
        },
        "gojourney": {
            "main": [
                config.challenge.prompt,
                ig_subnet.validator.get_promptGoJouney,
            ],
            "backup": [get_backup_prompt, ig_subnet.validator.get_promptGoJouney],
        },
        "text_generation": {
            "main": [config.challenge.llm_prompt],
            "backup": [get_backup_llm_prompt],
        },
        "controlnet": {
            "main": [config.challenge.prompt, config.challenge.image],
            "backup": [get_backup_prompt, get_backup_image],
        },
        "upscale": {"main": [config.challenge.image], "backup": [get_backup_image]},
        "ip_adapter": {
            "main": [
                config.challenge.prompt,
                config.challenge.image,
            ],
            "backup": [get_backup_prompt, get_backup_image],
        },
        "open_txt2img": {
            "main": [config.challenge.open_category_prompt],
            "backup": [get_backup_prompt],
        },
        "visual_question_answering": {
            "main": [config.challenge.visual_question_answering],
            "backup": [get_backup_challenge_vqa],
        },
    }
    return challenge_urls


def initialize_nicheimage_catalogue(config):
    nicheimage_catalogue = {
        "GoJourney": {
            "model_incentive_weight": 0.04,
            "supporting_pipelines": MODEL_CONFIGS["GoJourney"]["params"][
                "supporting_pipelines"
            ],
            "reward_url": ig_subnet.validator.get_reward_GoJourney,
            "reward_type": "custom_offline",
            "timeout": 12,
            "inference_params": {},
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "JuggernautXL": {
            "supporting_pipelines": MODEL_CONFIGS["JuggernautXL"]["params"][
                "supporting_pipelines"
            ],
            "model_incentive_weight": 0.07,
            "reward_url": config.reward_url.JuggernautXL,
            "reward_type": "image",
            "inference_params": {
                "num_inference_steps": 30,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 6,
            },
            "timeout": 12,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "RealitiesEdgeXL": {
            "supporting_pipelines": MODEL_CONFIGS["RealitiesEdgeXL"]["params"][
                "supporting_pipelines"
            ],
            "model_incentive_weight": 0.09,
            "reward_url": config.reward_url.RealitiesEdgeXL,
            "reward_type": "image",
            "inference_params": {
                "num_inference_steps": 7,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 5.5,
            },
            "timeout": 12,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "AnimeV3": {
            "supporting_pipelines": MODEL_CONFIGS["AnimeV3"]["params"][
                "supporting_pipelines"
            ],
            "model_incentive_weight": 0.09,
            "reward_url": config.reward_url.AnimeV3,
            "reward_type": "image",
            "inference_params": {
                "num_inference_steps": 25,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 7.0,
                "negative_prompt": "out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts",
            },
            "timeout": 12,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "Gemma7b": {
            "supporting_pipelines": MODEL_CONFIGS["Gemma7b"]["params"][
                "supporting_pipelines"
            ],
            "model_incentive_weight": 0.03,
            "timeout": 64,
            "synapse_type": ig_subnet.protocol.TextGenerating,
            "reward_url": config.reward_url.Gemma7b,
            "reward_type": "text",
            "inference_params": {},
        },
        "StickerMaker": {
            "supporting_pipelines": MODEL_CONFIGS["StickerMaker"]["params"][
                "supporting_pipelines"
            ],
            "model_incentive_weight": 0.03,
            "timeout": 64,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
            "reward_url": config.reward_url.StickerMaker,
            "reward_type": "image",
            "inference_params": {"is_upscale": False},
        },
        "Llama3_70b": {
            "supporting_pipelines": MODEL_CONFIGS["Llama3_70b"]["params"][
                "supporting_pipelines"
            ],
            "model_incentive_weight": 0.00,
            "timeout": 128,
            "synapse_type": ig_subnet.protocol.TextGenerating,
            "reward_url": config.reward_url.Llama3_70b,
            "reward_type": "text",
            "inference_params": {},
        },
        "Llama3_3_70b": {
            "supporting_pipelines": MODEL_CONFIGS["Llama3_3_70b"]["params"][
                "supporting_pipelines"
            ],
            "model_incentive_weight": 0.07,
            "timeout": 128,
            "synapse_type": ig_subnet.protocol.TextGenerating,
            "reward_url": config.reward_url.Llama3_3_70b,
            "reward_type": "text",
            "inference_params": {},
        },
        "SUPIR": {
            "supporting_pipelines": MODEL_CONFIGS["SUPIR"]["params"][
                "supporting_pipelines"
            ],
            "reward_url": config.reward_url.SUPIR,
            "reward_type": "image",
            "timeout": 180,
            "inference_params": {},
            "synapse_type": ig_subnet.protocol.ImageGenerating,
            "model_incentive_weight": 0.08,
        },
        "FluxSchnell": {
            "supporting_pipelines": MODEL_CONFIGS["FluxSchnell"]["params"][
                "supporting_pipelines"
            ],
            "model_incentive_weight": 0.12,
            "reward_url": config.reward_url.FluxSchnell,
            "reward_type": "image",
            "inference_params": {
                "num_inference_steps": 4,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 0.0,
            },
            "timeout": 24,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "Kolors": {
            "supporting_pipelines": MODEL_CONFIGS["Kolors"]["params"][
                "supporting_pipelines"
            ],
            "model_incentive_weight": 0.10,
            "reward_url": config.reward_url.Kolors,
            "reward_type": "image",
            "inference_params": {
                "num_inference_steps": 30,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 5.0,
            },
            "timeout": 32,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "OpenGeneral": {
            "supporting_pipelines": ["open_txt2img"],
            "model_incentive_weight": 0.08,
            "reward_url": config.reward_url.OpenCategory,
            "reward_type": "open_category",
            "inference_params": {},
            "timeout": 32,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "OpenDigitalArt": {
            "supporting_pipelines": ["open_txt2img"],
            "model_incentive_weight": 0.02,
            "reward_url": config.reward_url.OpenCategory,
            "reward_type": "open_category",
            "inference_params": {},
            "timeout": 32,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "OpenDigitalArtAnime": {
            "supporting_pipelines": ["open_txt2img"],
            "model_incentive_weight": 0.02,
            "reward_url": config.reward_url.OpenCategory,
            "reward_type": "open_category",
            "inference_params": {},
            "timeout": 32,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "OpenDigitalArtMinimalist": {
            "supporting_pipelines": ["open_txt2img"],
            "model_incentive_weight": 0.02,
            "reward_url": config.reward_url.OpenCategory,
            "reward_type": "open_category",
            "inference_params": {},
            "timeout": 32,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "OpenDigitalArtPixelArt": {
            "supporting_pipelines": ["open_txt2img"],
            "model_incentive_weight": 0.02,
            "reward_url": config.reward_url.OpenCategory,
            "reward_type": "open_category",
            "inference_params": {},
            "timeout": 32,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "OpenTraditionalArt": {
            "supporting_pipelines": ["open_txt2img"],
            "model_incentive_weight": 0.02,
            "reward_url": config.reward_url.OpenCategory,
            "reward_type": "open_category",
            "inference_params": {},
            "timeout": 32,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "OpenTraditionalArtPainting": {
            "supporting_pipelines": ["open_txt2img"],
            "model_incentive_weight": 0.02,
            "reward_url": config.reward_url.OpenCategory,
            "reward_type": "open_category",
            "inference_params": {},
            "timeout": 32,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "OpenTraditionalArtSketch": {
            "supporting_pipelines": ["open_txt2img"],
            "model_incentive_weight": 0.02,
            "reward_url": config.reward_url.OpenCategory,
            "reward_type": "open_category",
            "inference_params": {},
            "timeout": 32,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "OpenTraditionalArtComic": {
            "supporting_pipelines": ["open_txt2img"],
            "model_incentive_weight": 0.02,
            "reward_url": config.reward_url.OpenCategory,
            "reward_type": "open_category",
            "inference_params": {},
            "timeout": 32,
            "synapse_type": ig_subnet.protocol.ImageGenerating,
        },
        "Pixtral_12b": {
            "supporting_pipelines": ["visual_question_answering"],
            "model_incentive_weight": 0.04,
            "reward_url": config.reward_url.Pixtral_12b,
            "reward_type": "text",
            "inference_params": {
                "temperature": 0.7,
                "top_p": 1,
                "max_tokens": 8192,
                "logprobs": 100,
            },
            "timeout": 64,
            "synapse_type": ig_subnet.protocol.MultiModalGenerating,
        },
    }

    sum_incentive = 0
    for k, v in nicheimage_catalogue.items():
        sum_incentive += v["model_incentive_weight"]
    bt.logging.info(f"Sum incentive in code: {sum_incentive}")
    assert abs(sum_incentive - 1) < 1e-4

    return nicheimage_catalogue


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.challenge_urls = initialize_challenge_urls(self.config)
        self.nicheimage_catalogue = initialize_nicheimage_catalogue(self.config)
        self.open_category_reward_synapses = self.init_reward_open_category_synapses()
        self.miner_manager = MinerManager(self)
        self.load_state()
        self.update_scores_on_chain()
        self.sync()
        self.miner_manager.update_miners_identity()
        self.query_queue = QueryQueue(
            list(self.nicheimage_catalogue.keys()),
            time_per_loop=self.config.loop_base_time,
        )
        self.offline_reward = self.config.offline_reward.enable
        self.supporting_offline_reward_types = ["image", "custom_offline"]
        self.generate_response_offline_types = ["image"]
        if self.offline_reward:
            self.redis_client = RedisClient(
                url=self.config.offline_reward.redis_endpoint
            )
            self.reward_app = RewardApp(self)
            self.end_loop_event = threading.Event()
            threading.Thread(target=self.clear_data, daemon=True).start()
            threading.Thread(
                target=self.generate_validator_responses, daemon=True
            ).start()
            threading.Thread(target=self.reward_offline, daemon=True).start()

        if self.config.proxy.port:
            try:
                self.validator_proxy = ValidatorProxy(self)
                bt.logging.info("Validator proxy started succesfully")
            except Exception:
                bt.logging.warning(
                    "Warning, proxy did not start correctly, so no one can query through your validator. Error message: "
                    + traceback.format_exc()
                )

    def forward(self):
        """
        Validator synthetic forward pass. Consists of:
        - Querying all miners to get their model_name and total_volume
        - Create serving queue, here is pseudo code:
            ```
                synthentic_queue = Queue()
                for uid, total_volume_this_validator in all_uids_info:
                    for _ in range(total_volume_this_validator*0.8):
                        synthentic_queue.put(uid)
                shuffle(synthentic_queue)

                organic_queue = Queue()
                for uid, total_volume_this_validator in all_uids_info:
                    for _ in range(total_volume_this_validator*0.2):
                        organic_queue.put(uid)
                shuffle(organic_queue)
            ```
        - Forwarding requests to miners in multiple thread to ensure total time is around 600 seconds. In each thread, we do:
            - Calculating rewards if needed
            - Updating scores based on rewards
            - Saving the state
        - Normalize weights based on incentive_distribution
        - SET WEIGHTS!
        - Sleep for 600 seconds if needed
        """

        bt.logging.info("Updating available models & uids")
        async_batch_size = self.config.async_batch_size
        loop_base_time = self.config.loop_base_time
        self.open_category_reward_synapses = self.init_reward_open_category_synapses()
        threads = []
        loop_start = time.time()
        self.miner_manager.update_miners_identity()
        self.query_queue.update_queue(self.miner_manager.all_uids_info)
        self.rewarded_synapses = {model_name: [] for model_name in self.nicheimage_catalogue.keys()}
        self.not_rewarded_synapses = {model_name: [] for model_name in self.nicheimage_catalogue.keys()}

        for (
            model_name,
            uids,
            should_rewards,
            sleep_per_batch,
        ) in self.query_queue.get_batch_query(async_batch_size):
            bt.logging.info(
                f"Querying {len(uids)} uids for model {model_name}, sleep_per_batch: {sleep_per_batch}"
            )
            if model_name not in self.nicheimage_catalogue:
                bt.logging.warning(
                    f"Model {model_name} not in nicheimage_catalogue, skipping"
                )
                continue
            thread = threading.Thread(
                target=self.async_query_and_reward,
                args=(model_name, uids, should_rewards),
            )
            threads.append(thread)
            thread.start()

            bt.logging.info(f"Sleeping for {sleep_per_batch} seconds between batches")
            time.sleep(sleep_per_batch)

        for thread in threads:
            thread.join()

        bt.logging.info(
            "Loop completed, uids info:\n",
            str(self.miner_manager.all_uids_info).replace("},", "},\n"),
        )

        actual_time_taken = time.time() - loop_start

        bt.logging.debug(
            f"Open Synapse to be rewarded: {self.open_category_reward_synapses}"
        )

        if actual_time_taken < loop_base_time:
            bt.logging.info(
                f"Sleeping for {loop_base_time - actual_time_taken} seconds"
            )
            time.sleep(loop_base_time - actual_time_taken)

        if self.offline_reward:
            self.end_loop_event.set()

        self.update_scores_on_chain()
        self.save_state()

    def reward_offline(self):
        """Calculate rewards for miner based on  validator responses (from cache) and miner responses"""
        asyncio.get_event_loop().run_until_complete(
            self.reward_app.dequeue_reward_message()
        )

    def generate_validator_responses(self):
        """Handle generating validator responses for base synapses and cache the results to score the miner later"""
        asyncio.get_event_loop().run_until_complete(
            self.reward_app.dequeue_base_synapse_message()
        )

    def clear_data(self):
        """Process when the duration of one loop is complete."""
        while True:
            if self.end_loop_event.is_set():
                self.redis_client.get_stream_info(self.reward_app.reward_stream_name)
                self.redis_client.get_stream_info(
                    self.reward_app.base_synapse_stream_name
                )
                self.reward_app.show_total_uids_and_rewards()
                self.reward_app.reset_total_uids_and_rewards()
                self.end_loop_event.clear()
            time.sleep(10)

    def enqueue_synapse_for_validation(self, base_synapse):
        """Push base synapse to queue for generating validator response."""
        self.redis_client.publish_to_stream(
            stream_name=self.redis_client.base_synapse_stream_name,
            message={"data": json.dumps(base_synapse.deserialize())},
        )

    def async_query_and_reward(
        self,
        model_name: str,
        uids: list[int],
        should_rewards: list[int],
    ):
        start_async_query = time.time() # DEBUG
        dendrite = bt.dendrite(self.wallet)
        pipeline_type = random.choice(
            self.nicheimage_catalogue[model_name]["supporting_pipelines"]
        )
        reward_url = self.nicheimage_catalogue[model_name]["reward_url"]
        uids_should_rewards = list(zip(uids, should_rewards))
        synapses, batched_uids_should_rewards = self.prepare_challenge(
            uids_should_rewards, model_name, pipeline_type
        )

        def query_axons(axons, synapses):
            responses = [None] * len(axons)
            # Create event loop and run gather
            try:
                loop = asyncio.get_event_loop()
                responses = loop.run_until_complete(asyncio.gather(
                    *(dendrite.call(
                        axons, 
                        synapse, 
                        deserialize=False, 
                        timeout=self.nicheimage_catalogue[model_name]["timeout"]
                    ) for (axons, synapse) in zip(axons, synapses))
                ))
            except Exception:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                responses = new_loop.run_until_complete(asyncio.gather(
                    *(dendrite.call(
                        axons, 
                        synapse, 
                        deserialize=False, 
                        timeout=self.nicheimage_catalogue[model_name]["timeout"]
                    ) for (axons, synapse) in zip(axons, synapses))
                ))
                new_loop.close()
            finally:
                dendrite.close_session()
                return responses
            
        axons = []
        for uid in uids:
            if uid in self.miner_manager.layer_one_axons:
                axons.append(self.miner_manager.layer_one_axons[uid])
            else:
                axons.append(self.metagraph.axons[uid])
        start_query = time.time() # DEBUG
        _responses = query_axons(axons, synapses)
        end_query = time.time() # DEBUG
        bt.logging.info(f"Time taken for query: {end_query - start_query}") # DEBUG

        for synapse, uids_should_rewards, response in zip(synapses, batched_uids_should_rewards, _responses):
            start_loop = time.time() # DEBUG
            uids, should_rewards = zip(*uids_should_rewards)
            bt.logging.info(f"Quering {uids}, Should reward: {should_rewards}")
            if not synapse:
                continue
            base_synapse = synapse.model_copy()
            if (
                self.offline_reward
                and any([should_reward for should_reward in should_rewards])
                and self.nicheimage_catalogue[model_name]["reward_type"]
                in self.generate_response_offline_types
            ):
                self.enqueue_synapse_for_validation(base_synapse)

            axons = []
            for uid in uids:
                if uid in self.miner_manager.layer_one_axons:
                    axons.append(self.miner_manager.layer_one_axons[uid])
                else:
                    axons.append(self.metagraph.axons[uid])

            # responses = dendrite.query(
            #     axons=axons,
            #     synapse=synapse,
            #     deserialize=False,
            #     timeout=self.nicheimage_catalogue[model_name]["timeout"],
            # )
            responses = [response]
            reward_responses = [
                response
                for response, should_reward in zip(responses, should_rewards)
                if should_reward
            ]
            reward_uids = [
                uid for uid, should_reward in zip(uids, should_rewards) if should_reward
            ]

            bt.logging.info(
                f"Received {len(responses)} responses, {len(reward_responses)} to be rewarded"
            )
            store_thread = threading.Thread(
                target=self.store_miner_output,
                args=(self.config.storage_url, responses, uids),
                daemon=True,
            )
            start_store = time.time() # DEBUG
            store_thread.start()

            process_times = [
                synapse.dendrite.process_time if synapse.is_success else -1
                for synapse in responses
            ]
            self.miner_manager.update_metadata(uids, process_times)
            if reward_uids:
                if (
                    self.offline_reward
                    and self.nicheimage_catalogue[model_name]["reward_type"]
                    in self.supporting_offline_reward_types
                ):
                    ig_subnet.validator.get_reward_offline(
                        base_synapse,
                        reward_responses,
                        reward_uids,
                        self.nicheimage_catalogue[model_name].get("timeout", 12),
                        self.redis_client,
                    )
                else:
                    if callable(reward_url):
                        reward_uids, rewards = reward_url(
                            base_synapse, reward_responses, reward_uids
                        )
                    else:
                        start_reward = time.time() # DEBUG
                        reward_uids, rewards = ig_subnet.validator.get_reward(
                            reward_url,
                            base_synapse,
                            reward_responses,
                            reward_uids,
                            self.nicheimage_catalogue[model_name].get("timeout", 12),
                            self.miner_manager,
                        )
                        end_reward = time.time() # DEBUG
                        bt.logging.info(f"Time taken for reward: {end_reward - start_reward}") # DEBUG

                        # Scale Reward based on Miner Volume
                    for i, uid in enumerate(reward_uids):
                        if rewards[i] > 0:
                            rewards[i] = rewards[i] * (
                                0.6
                                + 0.4
                                * self.miner_manager.all_uids_info[uid]["reward_scale"]
                            )

                    bt.logging.info(f"Scored responses: {rewards}")

                    self.miner_manager.update_scores(reward_uids, rewards)
            store_thread.join()
            end_store = time.time() # DEBUG
            bt.logging.info(f"Time taken for store: {end_store - start_store}") # DEBUG

            end_loop = time.time() # DEBUG
            bt.logging.info(f"Time taken for loop: {end_loop - start_loop}") # DEBUG

        end_async_query = time.time() # DEBUG
        bt.logging.info(f"Time taken for async query: {end_async_query - start_async_query}") # DEBUG

    def prepare_challenge(self, uids_should_rewards, model_name, pipeline_type):
        """
        Batch the batch (max = 16) into smaller batch size (max = 4) and prepare synapses for each batch.
        """
        synapse_type = self.nicheimage_catalogue[model_name]["synapse_type"]
        model_miner_count = len(
            [
                uid
                for uid, info in self.miner_manager.all_uids_info.items()
                if info["model_name"] == model_name
            ]
        )
        # batch_size = min(4, 1 + model_miner_count // 4)
        batch_size = 1

        random.shuffle(uids_should_rewards)
        batched_uids_should_rewards = [
            uids_should_rewards[i * batch_size : (i + 1) * batch_size]
            for i in range((len(uids_should_rewards) + batch_size - 1) // batch_size)
        ]
        num_batch = len(batched_uids_should_rewards)
        synapses = [
            synapse_type(pipeline_type=pipeline_type, model_name=model_name)
            for _ in range(num_batch)
        ]
        for synapse in synapses:
            synapse.pipeline_params.update(
                self.nicheimage_catalogue[model_name]["inference_params"]
            )
            if self.nicheimage_catalogue[model_name]["reward_type"] == "open_category":
                width, height = random_image_size()
                synapse.pipeline_params.update({"width": width, "height": height})
            synapse.seed = random.randint(0, 1e9)
        for challenge_url, backup_func in zip(
            self.challenge_urls[pipeline_type]["main"],
            self.challenge_urls[pipeline_type]["backup"],
        ):
            if callable(challenge_url):
                synapses = challenge_url(synapses)
            else:
                assert isinstance(challenge_url, str)
                synapses = ig_subnet.validator.get_challenge(
                    challenge_url, synapses, backup_func
                )

        if self.nicheimage_catalogue[model_name]["reward_type"] != "open_category":
            for i, batch in enumerate(batched_uids_should_rewards):
                if any([should_reward for _, should_reward in batch]):
                    # select old rewarded synapse with probability
                    rand_val = random.random()
                    if len(self.rewarded_synapses[model_name]) > 0 and rand_val < 0.9:
                        if rand_val < 0.8:  # 80% chance to use existing synapse
                            synapses[i] = random.choice(self.rewarded_synapses[model_name]).model_copy(deep=True)
                        else: # 10% chance to use existing synapse with new seed
                            synapse = random.choice(self.rewarded_synapses[model_name]).model_copy(deep=True)
                            synapse.seed = random.randint(0, 1e9)
                            synapses[i] = synapse
                    else:
                        # else: 10% chance to use new synapse (already created)
                        self.rewarded_synapses[model_name].append(synapses[i].model_copy(deep=True))

                else:
                    # select old not rewarded synapse with probability
                    if random.random() < 0.1 and len(self.not_rewarded_synapses[model_name]) > 0:
                        synapses[i] = random.choice(self.not_rewarded_synapses[model_name]).model_copy(deep=True)
                    elif len(self.not_rewarded_synapses[model_name]) < len(self.rewarded_synapses[model_name]):
                        # limit the number of not rewarded synapses to be less or equal the number of rewarded synapses
                        self.not_rewarded_synapses[model_name].append(synapses[i].model_copy(deep=True))

        if self.nicheimage_catalogue[model_name]["reward_type"] == "open_category":
            # Reward same test for uids in same open category
            for i, batch in enumerate(batched_uids_should_rewards):
                if any([should_reward for _, should_reward in batch]):
                    self.open_category_reward_synapses[model_name] = (
                        self.open_category_reward_synapses[model_name] or synapses[i]
                    )
                    synapses[i] = self.open_category_reward_synapses[model_name]

        return synapses, batched_uids_should_rewards

    def store_miner_output(
        self, storage_url, responses: list[bt.Synapse], uids
    ):
        if not self.config.share_response:
            return

        for uid, response in zip(uids, responses):
            if not response.is_success:
                continue
            try:
                response.store_response(storage_url, uid, self.uid, self.wallet.hotkey)
                break
            except Exception as e:
                bt.logging.error(f"Error in storing response: {e}")

    def init_reward_open_category_synapses(self):
        """
        Initialize synapses to be rewarded for open category models.
        """
        return {
            k: None
            for k in self.nicheimage_catalogue.keys()
            if self.nicheimage_catalogue[k]["reward_type"] == "open_category"
        }

    def update_scores_on_chain(self):
        """
        Update weights based on incentive pool and model specific weights.
        - Apply rank weight for open category model.
        """
        weights = np.zeros(len(self.miner_manager.all_uids))

        for model_name in self.nicheimage_catalogue.keys():
            model_specific_weights = self.miner_manager.get_model_specific_weights(model_name)
            if self.nicheimage_catalogue[model_name]["reward_type"] == "open_category":
                mask = model_specific_weights > 1e-4
                ranked_model_specific_weights = self.rank_array(model_specific_weights)
                bt.logging.debug(
                    f"Unique ranked weights for {model_name}\n{np.unique(model_specific_weights)}"
                )
                model_specific_weights = (
                    model_specific_weights * 0.5 + ranked_model_specific_weights * 0.5
                )
                model_specific_weights = 0.8 + 0.2 * model_specific_weights
                model_specific_weights = model_specific_weights * mask
                raw_weight_sum = np.sum(np.abs(model_specific_weights), axis=0, keepdims=True)
                if not raw_weight_sum == 0:
                    model_specific_weights = model_specific_weights / raw_weight_sum

                bt.logging.debug(f"Normalized {model_name} weights\n{model_specific_weights}")
            # Smoothing update incentive
            temp_incentive_weight = {}
            if datetime.utcnow() < datetime(2024, 12, 12, 14, 0, 0):
                temp_incentive_weight = {
                    "RealitiesEdgeXL": 0.13,
                    "AnimeV3": 0.12,
                    "JuggernautXL": 0.10,
                    "Llama3_3_70b": 0.00,
                    "Llama3_70b": 0.05,
                    "OpenDigitalArt": 0.08,
                    "OpenDigitalArtAnime": 0.00,
                    "OpenDigitalArtMinimalist": 0.00,
                    "OpenDigitalArtPixelArt": 0.00,
                    "OpenTraditionalArt": 0.00,
                    "OpenTraditionalArtPainting": 0.00,
                    "OpenTraditionalArtSketch": 0.00,
                    "OpenTraditionalArtComic": 0.00,
                }
            elif datetime.utcnow() < datetime(2024, 12, 19, 14, 0, 0):
                temp_incentive_weight = {
                    "RealitiesEdgeXL": 0.115,
                    "AnimeV3": 0.11,
                    "JuggernautXL": 0.095,
                    "Llama3_3_70b": 0.025,
                    "Llama3_70b": 0.025,
                    "OpenDigitalArt": 0.05,
                    "OpenDigitalArtAnime": 0.01,
                    "OpenDigitalArtMinimalist": 0.01,
                    "OpenDigitalArtPixelArt": 0.01,
                    "OpenTraditionalArt": 0.0075,
                    "OpenTraditionalArtPainting": 0.0075,
                    "OpenTraditionalArtSketch": 0.0075,
                    "OpenTraditionalArtComic": 0.0075,
                }
            elif datetime.utcnow() < datetime(2024, 12, 26, 14, 0, 0):
                temp_incentive_weight = {
                    "RealitiesEdgeXL": 0.10,
                    "AnimeV3": 0.10,
                    "JuggernautXL": 0.09,
                    "Llama3_3_70b": 0.03,
                    "Llama3_70b": 0.02,
                    "OpenDigitalArt": 0.035,
                    "OpenDigitalArtAnime": 0.015,
                    "OpenDigitalArtMinimalist": 0.015,
                    "OpenDigitalArtPixelArt": 0.015,
                    "OpenTraditionalArt": 0.015,
                    "OpenTraditionalArtPainting": 0.015,
                    "OpenTraditionalArtSketch": 0.015,
                    "OpenTraditionalArtComic": 0.015,
                }
            elif datetime.utcnow() < datetime(2025, 1, 30, 16, 0, 0):
                # Before first reduction - original weights
                temp_incentive_weight = {
                    "RealitiesEdgeXL": 0.09,
                    "AnimeV3": 0.09,
                }
            elif datetime.utcnow() < datetime(2025, 2, 3, 16, 0, 0):
                # After first reduction - 50% weights
                temp_incentive_weight = {
                    "RealitiesEdgeXL": 0.045,
                    "AnimeV3": 0.045,
                }
            else:
                # After second reduction - 0% weights
                temp_incentive_weight = {
                    "RealitiesEdgeXL": 0.0,
                    "AnimeV3": 0.0,
                }
            # TODO: after we updated weights, we need to update the model_incentive_weight in the miner_manager
            if model_name in temp_incentive_weight:
                bt.logging.info(
                    f"Using temp_incentive_weight: {temp_incentive_weight} for {model_name}"
                )
                model_specific_weights = (
                    model_specific_weights * temp_incentive_weight[model_name]
                )
            else:
                model_specific_weights = (
                    model_specific_weights
                    * self.nicheimage_catalogue[model_name]["model_incentive_weight"]
                )
            bt.logging.info(
                f"model_specific_weights for {model_name}\n{model_specific_weights}"
            )

            weights = weights + model_specific_weights

        # Check if rewards contains NaN values.
        if np.isnan(weights).any():
            bt.logging.warning(f"NaN values detected in weights: {weights}")
            # Replace any NaN values in rewards with 0.
            weights = np.nan_to_num(weights, nan=0)

        self.scores: np.ndarray = weights
        bt.logging.success(f"Updated scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file using pickle."""
        state = {
            "step": self.step,
            "all_uids_info": self.miner_manager.all_uids_info,
        }
        try:
            # Open the file in write-binary mode
            with open(self.config.neuron.full_path + "/state.pkl", "wb") as f:
                pickle.dump(state, f)
            bt.logging.info("State successfully saved to state.pkl")
        except Exception as e:
            bt.logging.error(f"Failed to save state: {e}")


    def load_state(self):
        """Loads the state of the validator from a file, with fallback to .pt if .pkl is not found."""
        # TODO: After a transition period, remove support for the old .pt format.
        try:
            path_pt = self.config.neuron.full_path + "/state.pt"
            path_pkl = self.config.neuron.full_path + "/state.pkl"

            # Try to load the newer .pkl format first
            try:
                bt.logging.info(f"Loading validator state from: {path_pkl}")
                with open(path_pkl, "rb") as f:
                    state = pickle.load(f)

                # Restore state from pickle file
                self.step = state["step"]
                self.miner_manager.all_uids_info = state["all_uids_info"]
                bt.logging.info("Successfully loaded state from .pkl file")
                return  # Exit after successful load from .pkl

            except Exception as e:
                bt.logging.warning(f"Failed to load from .pkl format: {e}")

            # If .pkl loading fails, try to load from the old .pt file (PyTorch format)
            try:
                bt.logging.info(f"Loading validator state from: {path_pt}")
                state = torch.load(path_pt)

                # Restore state from .pt file
                self.step = state["step"]
                self.miner_manager.all_uids_info = state["all_uids_info"]
                bt.logging.info("Successfully loaded state from .pt file")

            except Exception as e:
                bt.logging.error(f"Failed to load from .pt format: {e}")
                self.step = 0  # Default fallback when both load attempts fail
                bt.logging.error("Could not find previously saved state or error loading it.")

        except Exception as e:
            self.step = 0  # Default fallback in case of an unknown error
            bt.logging.error(f"Error loading state: {e}")


    @staticmethod
    def rank_array(array: np.ndarray):
        # Return Zeros if array is zeros
        if np.sum(array) == 0:
            return array
        # Step 1: Sort the array and get the original indices
        sorted_array = np.sort(array)[::-1]  # Sort in descending order
        indices = np.argsort(array)[::-1]  

        # Step 2: Create a new array for rankings
        ranked_array = np.zeros_like(array)

        # Step 3: Assign ranks based on conditions
        # First element gets 1.0
        ranked_array[indices[0]] = 1.0

        # Check for tie between second and third elements
        if sorted_array[1] == sorted_array[2]:
            # If there's a tie, both get 0.5
            ranked_array[indices[1]] = 0.5
            ranked_array[indices[2]] = 0.5
        else:
            # Otherwise, assign 2/3 and 1/3
            ranked_array[indices[1]] = 2 / 3
            ranked_array[indices[2]] = 1 / 3

        # All others (rank 4 and below) get 0 (already initialized)

        return ranked_array


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(360)