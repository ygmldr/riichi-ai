"""libriichi 冒烟测试:常量、obs 形状、单局解析。"""
import os
import sys
import zipfile
import gzip

# 将构建产物加入 path(与 build_libriichi.sh 一致)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "vendor", "mortal", "target", "debug"))

import libriichi  # noqa: E402
from libriichi.consts import ACTION_SPACE, GRP_SIZE, obs_shape  # noqa: E402


def test_consts():
    assert ACTION_SPACE == 46
    assert GRP_SIZE == 7
    assert obs_shape(4) == (1012, 34)


def test_parse_one_game():
    """用 2009.zip 里的一局验证 GameplayLoader 能解析 .mjson(gzip)。"""
    zip_path = os.environ.get("TEST_ZIP", "/tmp/mjai-test/2009.zip")
    if not os.path.exists(zip_path):
        return  # 数据未下载时跳过
    loader = libriichi.dataset.GameplayLoader(version=4, oracle=False)
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        raw = z.read(name)  # .mjson 是 gzip 压缩的
        text = gzip.decompress(raw).decode("utf-8")
    games = loader.load_log(text)
    assert len(games) == 4  # 四人麻将,4 个玩家视角
    obs = games[0].take_obs()  # 每步一个 PyArray2:list[np.ndarray]
    assert len(obs) > 0, obs
    assert obs[0].shape[1] == 34, obs[0].shape


if __name__ == "__main__":
    test_consts()
    test_parse_one_game()
    print("libriichi 冒烟测试全部通过")
