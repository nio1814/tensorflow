# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Functions to create a Markov Chain Monte Carlo Metropolis step."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.contrib.bayesflow.python.ops.metropolis_hastings_impl import *
# pylint: enable=wildcard-import
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'evolve',
    'uniform_random_proposal',
    'normal_random_proposal',
]

remove_undocumented(__name__, _allowed_symbols)
