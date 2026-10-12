# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""eval/unitxt on skypilot/aws with one GPU (real EC2).

The GPU counterpart of ``test/slurm-cpu/test_skypilot_slurm_unitxt_cpu.py``. One
target evaluates the step's documented example, ``ibm-granite/granite-4.2-3b`` (bound
as the target's ``model`` input), on ten ``cards.mmlu_pro.engineering`` instances on
one A10G. The score is printed by unitxt into the job log but not asserted; reaching
SUCCESS proves:

* the ``hf://`` model input is downloaded inline into the step's instance and its
  path reaches ``--model_args pretrained=...``;
* setup's default torch (PyPI's Linux wheel, which bundles CUDA) works with the
  node's NVIDIA driver, and ``device=cuda`` runs;
* ``unitxt-evaluate --model hf`` completes on a GPU and ``results`` is registered.

**Real EC2, billed: it never runs by accident.** It is extended-suite only and skips
unless AWS credentials are present
(:func:`gbserver.environment.skypilot.aws_credentials_present`), so no instance is
provisioned without credentials explicitly exported. The fixture runs only the
happy path, with no injected failure, so one run is one EC2 launch.

The model and the dataset are public, so no HF_TOKEN is needed. The output is
``env://``, so nothing is pushed anywhere.

Fixtures (build.yaml/buildtest.yaml) live in the ``test-data/`` dir mirroring this
file, resolved by the repo's ``test/`` <-> ``test-data/`` helper so the pairing holds
in both test modes (see steps/README.md).
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

from gbserver.environment.skypilot import aws_credentials_present

pytestmark = pytest.mark.skypilot_integration


@extended_testing_only
@pytest.mark.skipif(
    not aws_credentials_present(),
    reason=(
        "AWS credentials not in environment "
        "(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)"
    ),
)
class TestSkypilotAwsUnitxtGpu(AbstractYamlBuildRunnerTest):
    """eval/unitxt step: hf mode on one A10G, end to end on aws (EC2)."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
