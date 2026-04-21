from src.common.config import LANGJEPAConfig
from src.decoder.config import DecoderFullConfig


def test_encoder_yaml_loads_cleanly():
    cfg = LANGJEPAConfig.from_yaml("src/encoder/configs/base_lang_config.yaml")
    # Sanity checks on the new Phase-1 fields.
    assert cfg.model.pretrained is True
    assert cfg.optimization.loss_fn == "smooth_l1"
    assert 0.0 < cfg.optimization.momentum_start <= cfg.optimization.momentum_end <= 1.0
    assert cfg.data.window_size >= 1
    # pred_dim must match embed_dim in our I-JEPA-style setup.
    assert cfg.model.pred_dim == cfg.model.embed_dim


def test_decoder_yaml_loads_cleanly():
    cfg = DecoderFullConfig.from_yaml("src/decoder/configs/decoder_config.yaml")
    assert cfg.decoder.hidden_dim > 0
    assert cfg.training.batch_size > 0
    assert 0.0 <= cfg.training.eval_ratio < 1.0
    assert cfg.evaluation.eval_steps > 0


def test_encoder_config_round_trip_via_yaml(tmp_path):
    cfg = LANGJEPAConfig.from_yaml("src/encoder/configs/base_lang_config.yaml")
    cfg.to_yaml(str(tmp_path / "round.yaml"))
    cfg2 = LANGJEPAConfig.from_yaml(str(tmp_path / "round.yaml"))
    assert cfg.model.pretrained == cfg2.model.pretrained
    assert cfg.optimization.lr == cfg2.optimization.lr
