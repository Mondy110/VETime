"""统一评估/推理启动入口。"""

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.seed import seed_everything
from src.utils.logger import get_logger

log = get_logger(__name__)


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    log.info("VETime 评估启动")
    seed_everything(cfg.seed)

    # TODO: 构建模型、加载权重、创建 Evaluator、运行评估
    # 完整实现需要与 train_univariate_hydra 的模型构建逻辑对齐
    log.info("评估完成 (skeleton — 完整实现待后续迭代)")


if __name__ == "__main__":
    main()
