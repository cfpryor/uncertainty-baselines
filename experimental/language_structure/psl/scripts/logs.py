# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

# Lint as: python3

"""Logging util."""

import logging

def initLogging(logging_level = logging.INFO):
    """
    Initializes the logging format and level.
    """

    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging_level)

def updateLoggingLevel(logging_level):
    """
    Updates the logging level.
    """

    logger = logging.getLogger()
    logger.setLevel(logging_level)