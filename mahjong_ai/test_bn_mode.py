"""
Test : une policy BatchNorm en mode train depend de la taille du batch.

Montre le bug corrige par OracleDiscardAgent.__init__ (policy.eval()) :
en mode train, les logits d'un meme etat different selon qu'il est evalue
seul (rollout, batch de 1) ou dans un batch (update PPO). En eval, ils sont
identiques.

Aucun checkpoint requis (petit modele aleatoire).

Usage :
    py -3.14 -m mahjong_ai.test_bn_mode
"""
import torch

from mahjong_ai.agents.suphx_cnn import build_model, IN_CHANNELS_ORACLE


def max_gap(model, x):
    """Ecart max entre logits calcules un par un et logits en batch."""
    with torch.no_grad():
        one_by_one = torch.cat([model(x[i:i + 1]) for i in range(len(x))])
        batched = model(x)
    return (one_by_one - batched).abs().max().item()


def main():
    torch.manual_seed(0)
    model = build_model('discard', use_oracle=True, channels=32, n_blocks=3)

    # Stats BN non triviales (comme apres un SL) : quelques gros batchs.
    model.train()
    with torch.no_grad():
        for _ in range(20):
            model(torch.randn(256, IN_CHANNELS_ORACLE, 34))

    x = torch.randn(16, IN_CHANNELS_ORACLE, 34)

    model.train()
    gap_train = max_gap(model, x)
    model.eval()
    gap_eval = max_gap(model, x)

    print(f"mode train : ecart max batch-1 vs batch-16 = {gap_train:.4f}")
    print(f"mode eval  : ecart max batch-1 vs batch-16 = {gap_eval:.2e}")

    assert gap_train > 1e-2, "attendu : logits dependants du batch en train"
    assert gap_eval < 1e-4, "attendu : logits identiques en eval"
    print("OK : la policy doit etre en eval() pour le self-play.")


if __name__ == '__main__':
    main()
