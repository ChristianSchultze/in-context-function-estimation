from pathlib import Path

from icfelab.sample import save_compressed_json
from icfelab.utils import read_cepheid

data = read_cepheid()
# plot_raw(data[:, 0], data[:, 1], Path("cepheid"))

result_list = []
for i in range(100):
    input_data={"indices": data[:, 0][i:i+50].tolist(), "values": data[:, 1][i:i+50].tolist()}
    result_list.append({"target": [0] * 128, "input": input_data, "rbf_scale": 0})

save_compressed_json(result_list, Path("cepheid.lzma"))
