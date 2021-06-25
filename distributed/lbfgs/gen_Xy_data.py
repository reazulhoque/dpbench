# Copyright (C) 2021 Intel corporation
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

import numpy as np
shape = (5000, 100)
rs = np.random.RandomState(1337)
X1 = rs.normal(loc=5.0, size=shape)
y1 = np.zeros(shape=(shape[0],), dtype=int)
X2 = rs.normal(loc=10.0, size=shape)
y2 = np.ones(shape=(shape[0],), dtype=int)
X = np.concatenate([X1, X2], axis=0)
y = np.concatenate([y1, y2], axis=0)
np.savetxt("X.csv", X, delimiter=',')
np.savetxt("y.csv", y, delimiter=',')
print(y.shape)
