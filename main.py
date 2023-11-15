import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from encortex import Consumer, Contract, Market, Producer, Source
from encortex.markets import DAM
from encortex.sources import Solar, Wind


@hydra.main(config_path="conf", config_name="config")
def run_encortex(cfg: DictConfig) -> None:
    producer: Producer = instantiate(cfg.producer)
    assert isinstance(
        producer, Producer
    ), f"Wrong object instatiated: {type(producer)}"

    contracts_cfg = cfg.contracts
    contracts = []
    for contract_config in contracts_cfg.keys():
        contract: Contract = instantiate(contracts_cfg[contract_config])
        assert isinstance(
            contract, Contract
        ), f"Wrong object instantiated: {type(contract)}"
        contracts.append(contract)
        producer.add(contract)

    decision_units = producer.get_decision_units()


if __name__ == "__main__":
    run_encortex()
