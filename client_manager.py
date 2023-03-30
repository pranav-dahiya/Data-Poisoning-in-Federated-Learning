#!/usr/bin/env python
# -*-coding:utf-8 -*-

# @File    :   client.py
# @Time    :   2023/01/21 11:36:46
# @Author  :   Alexandru-Andrei Iacob
# @Contact :   aai30@cam.ac.uk
# @Author  :   Lorenzo Sani
# @Contact :   ls985@cam.ac.uk, lollonasi97@gmail.com
# @Version :   1.0
# @License :   (C)Copyright 2023, Alexandru-Andrei Iacob, Lorenzo Sani
# @Desc    :   None

from typing import Optional, List
import random

from logging import INFO, WARNING

from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class CustomClientManager(SimpleClientManager):
    def __init__(self, criterion: Criterion, seed: int) -> None:
        super().__init__()
        self.criterion = criterion
        self.seed = seed

    def sample(
        self,
        num_clients: int,
        server_round: int,
        min_num_clients: Optional[int] = None,
        current_virtual_clock: Optional[float] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        cids = list(self.clients)
        # Shuffle the list of clients
        random.seed(self.seed)
        for _ in range(server_round):
            random.shuffle(cids)
        log(INFO, f"Sampling using {self.criterion}")
        available_cids = []
        if self.criterion is not None:
            self.criterion.current_virtual_clock = current_virtual_clock  # type: ignore
            while len(available_cids) < num_clients and len(cids) > 0:
                cid = cids.pop()
                if self.criterion.select(self.clients[cid]):
                    available_cids.append(cid)
        else:
            available_cids = random.sample(cids, num_clients)

        if num_clients > len(available_cids):
            log(
                WARNING,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        client_list = [self.clients[cid] for cid in available_cids]
        print("Sampled the following clients: ", available_cids)
        return client_list
