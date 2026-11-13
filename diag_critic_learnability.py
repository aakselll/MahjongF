"""[V2.2] Expérience décisive : le critic peut-il prédire G_t OUT-OF-SAMPLE ?

Contexte (run v2.1, checkpoints/rl_v21, 265k steps arrêté) : la fenêtre de
replay a réglé « le critic ne voit jamais deux fois les mêmes données » mais
produit l'échec inverse — le critic MÉMORISE la fenêtre :
    cfit 0.104 -> 0.024 (10x sous le prédicteur-constant ~0.25)
    EV sur batch frais : +0.000 -> -0.44 (prédictions confiantes non
    corrélées : Var(A) = Var(G) + Var(V))
    porte min_ev : 122 rejets sur 122 updates -> batch_mean partout.

Cette expérience répond AVANT de relancer un run de 8h :

  1. collecte N batches SANS entraînement (policy = SL figée, conditions
     du trainer : mode=opponent, gamma_t=0, danger labels) ;
  2. split temporel train / val / test — le test n'est JAMAIS vu, il joue
     le rôle du « batch frais » que mesure la porte EV du trainer ;
  3. pour chaque config (weight_decay x archi) : critic NEUF, entraîné
     sur train, époque sélectionnée sur val (early-stop), mesuré sur
     test : EV = 1 - Var(G - V)/Var(G), exactement la métrique de la
     porte (min_ev=0.02) ;
  4. références : prédicteur-constant (EV=0 par construction) et
     moyenne-du-train (montre le coût du décalage batch).

Verdicts :
    EV test >= 0.10  -> relancer le trainer avec --critic-wd <gagnant>
    0.02 <= EV < 0.10 -> marginal (la porte passera rarement) ;
    EV <= 0.02       -> cible non prédictible par cette archi : NE PAS
                        relancer en --adv-mode critic ; rapporter le
                        tableau (plan B : entrée features GRP, archi
                        différente).

Exécution (~1h : collecte ~50 min + ~10 configs x ~30s de fit) :
    py -3.14 -m mahjong_ai.diag_critic_learnability \
        --sl-discard checkpoints/sl_tonpusen_la/discard.pt \
        --sl-frozen checkpoints/sl_tonpusen_la \
        --grp-checkpoint checkpoints/grp_tonpusen.pt \
        --compute-lookahead
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong_ai.rl.agent import PPOAgent
from mahjong_ai.rl.rollout import VecRunner
from mahjong_ai.agents.oracle_agent import Critic


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.
                                RawDescriptionHelpFormatter)
    # mêmes chemins que rl.train
    p.add_argument('--sl-discard', required=True)
    p.add_argument('--sl-frozen', required=True)
    p.add_argument('--grp-checkpoint', default=None)
    # collecte : conditions identiques au trainer
    p.add_argument('--n-envs', type=int, default=16)
    p.add_argument('--batch-games', type=int, default=32)
    p.add_argument('--n-batches', type=int, default=20,
                   help='total collecté (= 12 train + 4 val + 4 test '
                        'par défaut)')
    p.add_argument('--train-batches', type=int, default=12)
    p.add_argument('--val-batches', type=int, default=4)
    p.add_argument('--compute-lookahead', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--torch-threads', type=int, default=2)
    # sweep
    p.add_argument('--wd-list', type=str, default='0,1e-4,3e-4,1e-3,3e-3')
    p.add_argument('--archi-list', type=str, default='64x4,32x2')
    p.add_argument('--steps', type=int, default=4000,
                   help='budget max par config (early-stop sur val)')
    p.add_argument('--minibatch', type=int, default=512)
    p.add_argument('--check-every', type=int, default=25)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--critic-lr', type=float, default=1e-3)
    p.add_argument('--reward-scale', type=float, default=10.0)
    return p


# ---------------------------------------------------------------------- #
def _forward(critic, obs_n, obs_of, device, rs):
    """V(s) en unités reward, par tranches."""
    out = []
    with torch.no_grad():
        for s in range(0, len(obs_n), 2048):
            on = torch.as_tensor(obs_n[s:s + 2048], dtype=torch.float32,
                                 device=device)
            oof = torch.as_tensor(obs_of[s:s + 2048], dtype=torch.float32,
                                  device=device)
            out.append(critic(torch.cat([on, oof], dim=1)).cpu().numpy())
    return np.concatenate(out) * rs


def _eval_loss(critic, obs_n, obs_of, rets, device, rs):
    """0.5*MSE en unités normalisées (comparable à vl / cfit)."""
    v = _forward(critic, obs_n, obs_of, device, rs) / rs
    return float(0.5 * ((v - rets / rs) ** 2).mean())


def fit_one(train, val, test_batches, wd, archi, args, device):
    """Un critic neuf -> époque choisie sur val -> mesures sur test."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    critic = Critic(channels=archi[0], n_blocks=archi[1]).to(device)
    critic.eval()
    opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    rng = np.random.default_rng(args.seed)

    tn = np.stack([t['obs_normal'] for t in train]).astype(np.float16)
    to = np.stack([t['obs_oracle_full'] for t in train]).astype(np.float16)
    tr = np.array([t['return'] for t in train], dtype=np.float32)
    vn = np.stack([t['obs_normal'] for t in val]).astype(np.float16)
    vo = np.stack([t['obs_oracle_full'] for t in val]).astype(np.float16)
    vr = np.array([t['return'] for t in val], dtype=np.float32)

    best_val, best_state, since_best, best_step = None, None, 0, 0
    n = len(train)
    for it in range(1, args.steps + 1):
        idx = rng.integers(0, n, size=args.minibatch)
        on = torch.as_tensor(tn[idx], dtype=torch.float32, device=device)
        oof = torch.as_tensor(to[idx], dtype=torch.float32, device=device)
        ret = torch.as_tensor(tr[idx], dtype=torch.float32,
                              device=device) / args.reward_scale
        v = critic(torch.cat([on, oof], dim=1))
        loss = 0.5 * ((v - ret) ** 2).mean()
        if wd > 0:
            loss = loss + wd * sum(torch.sum(p * p)
                                   for p in critic.parameters())
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % args.check_every == 0:
            vloss = _eval_loss(critic, vn, vo, vr, device, args.reward_scale)
            if best_val is None or vloss < best_val - 1e-6:
                best_val, since_best, best_step = vloss, 0, it
                best_state = {k: w.detach().cpu().clone()
                              for k, w in critic.state_dict().items()}
            else:
                since_best += 1
                if since_best >= args.patience:
                    break
    if best_state is not None:
        critic.load_state_dict(best_state)

    evs, losses = [], []
    for tb in test_batches:
        tn_b = np.stack([t['obs_normal'] for t in tb]).astype(np.float16)
        to_b = np.stack([t['obs_oracle_full'] for t in tb]).astype(np.float16)
        G = np.array([t['return'] for t in tb], dtype=np.float32)
        v = _forward(critic, tn_b, to_b, device, args.reward_scale)
        ev = 1.0 - float((G - v).var()) / max(float(G.var()), 1e-8)
        evs.append(ev)
        losses.append(float(0.5 * ((v / args.reward_scale - G
                                    / args.reward_scale) ** 2).mean()))
    return {'best_val': best_val, 'best_step': best_step,
            'evs': evs, 'ev_mean': float(np.mean(evs)),
            'ev_min': float(np.min(evs)), 'losses': losses}


# ---------------------------------------------------------------------- #
def main():
    args = build_parser().parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"[diag] GPU : {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_num_threads(args.torch_threads)
    print(f"[diag] protocole : {args.n_batches} batches ({args.train_batches}"
          f" train / {args.val_batches} val / "
          f"{args.n_batches - args.train_batches - args.val_batches} test), "
          f"policy = SL figée, gamma_t=0, mode=opponent", flush=True)

    agent = PPOAgent(args.sl_discard, args.sl_frozen, device=device,
                     critic_lr=args.critic_lr)
    runner = VecRunner(
        agent, n_envs=args.n_envs, seed=args.seed,
        grp_path=args.grp_checkpoint,
        compute_lookahead=args.compute_lookahead, danger_labels=True,
        mode='opponent', device=device)

    # ---------------- collecte ---------------- #
    batches = []
    t0 = time.time()
    for i in range(args.n_batches):
        eps, _ = runner.collect_batch(args.batch_games, gamma_t=0.0)
        trs = [t for ep in eps for t in ep]
        rets = np.array([t['return'] for t in trs], dtype=np.float32)
        n_distinct = len(np.unique(np.round(rets, 3)))
        batches.append(trs)
        print(f"  [{i+1:2d}/{args.n_batches}] {len(trs)} tr, "
              f"{n_distinct} cibles distinctes (G constant/manche), "
              f"mean={rets.mean():+.2f} var={rets.var():.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    n_tr, n_va = args.train_batches, args.val_batches
    test_batches = batches[n_tr + n_va:]
    assert len(test_batches) >= 2, "pas assez de batches de test"
    train = [t for b in batches[:n_tr] for t in b]
    val = [t for b in batches[n_tr:n_tr + n_va] for t in b]
    print(f"[diag] splits : train={len(train)} val={len(val)} "
          f"test={sum(len(b) for b in test_batches)} tr", flush=True)

    # références sur test
    print("\n[diag] références (test) :")
    train_mean = float(np.mean([t['return'] for t in train]))
    for k, tb in enumerate(test_batches):
        G = np.array([t['return'] for t in tb], dtype=np.float32)
        ev_const = 0.0   # par construction
        ev_trmean = 1.0 - float((G - train_mean).var()) / max(G.var(), 1e-8)
        loss_const = float(0.5 * G.var()) / args.reward_scale ** 2
        print(f"  test[{k}] : n={len(tb)}, Var(G)={G.var():.1f}, "
              f"EV(const)= {ev_const:+.3f}, EV(moy-train)={ev_trmean:+.3f}, "
              f"loss(const)={loss_const:.3f}", flush=True)

    # ---------------- sweep ---------------- #
    wds = [float(x) for x in args.wd_list.split(',')]
    archis = []
    for a in args.archi_list.split(','):
        c, b = a.split('x')
        archis.append((int(c), int(b)))
    print(f"\n[diag] sweep : wd x {wds} | archi x {archis} | "
          f"budget {args.steps} steps, early-stop (val, patience "
          f"{args.patience})\n", flush=True)

    results = []
    for archi in archis:
        for wd in wds:
            r = fit_one(train, val, test_batches, wd, archi, args, device)
            results.append((wd, archi, r))
            print(f"  wd={wd:<6g} archi={archi[0]}x{archi[1]} : "
                  f"val={r['best_val']:.3f}@{r['best_step']} steps | "
                  f"EV test min/mean = {r['ev_min']:+.3f}/{r['ev_mean']:+.3f}"
                  f" | loss test {np.mean(r['losses']):.3f}", flush=True)

    # ---------------- verdict ---------------- #
    best = max(results, key=lambda x: x[2]['ev_mean'])
    wd_b, archi_b, r_b = best
    print("\n" + "=" * 64)
    print(f"[VERDICT] meilleure config : wd={wd_b:g} archi={archi_b[0]}x"
          f"{archi_b[1]} -> EV test moyen {r_b['ev_mean']:+.3f} "
          f"(min {r_b['ev_min']:+.3f})")
    if r_b['ev_mean'] >= 0.10:
        print("[VERDICT] LE CRITIC GÉNÉRALISE. Relancer le trainer avec :")
        print(f"          --critic-wd {wd_b:g}   (garder --adv-mode critic "
              f"et le replay v2.1)")
        print("          lecture attendue : part (critic) > 70%, ev > +0.10,")
        print("          cfit ~ cval, agr stabilise >= 0.90, evaux a la "
              "baisse d'ici 100k steps.")
    elif r_b['ev_mean'] >= 0.02:
        print("[VERDICT] MARGINAL : la porte min_ev=0.02 passera rarement.")
        print("          Option : relancer quand même et surveiller la part "
              "(critic) ;")
        print("          sinon attendre et rapporter ce tableau.")
    else:
        print("[VERDICT] NON-PRÉDICTIBLE out-of-sample par ce critic.")
        print("          NE PAS relancer en --adv-mode critic (la porte "
              "rejetterait")
        print("          encore tout, cf. run v2.1). Rapporter ce tableau :")
        print("          plan B = features GRP en entrée du critic, ou "
              "changer la cible")
        print("          (delta de manche plutôt que G_t) — mais d'abord "
              "mesurer, pas deviner.")
    print("=" * 64)


if __name__ == '__main__':
    main()
