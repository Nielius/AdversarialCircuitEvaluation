from uuid import uuid4

from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName
from acdc.nudb.adv_opt.experiments.core.k8s_launcher import launch
from acdc.nudb.adv_opt.experiments.w15_0_command import command

for task in [AdvOptTaskName.IOI, AdvOptTaskName.GREATERTHAN]:
    for rs in list(range(4312, 4313)):
        launch(
            job_name=f"test-renorm-{task.lower()}-{rs}-{uuid4().hex[:8]}",
            command=command(random_seed=rs, task=task),
            task=task,
            commit_hash="niels/wip",
            should_follow_logs=(rs == 4331 and task == AdvOptTaskName.GREATERTHAN),
        )
