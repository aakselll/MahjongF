"""[V2.3] Le critic peut-il prédire la valeur ? Deuxième salve de sondes.

Verdict v2.2 (expérience précédente, policy SL figée, 20 batches,
26.5k transitions) : NON-PRÉDICTIBLE out-of-sample. EV test +0.008 au
mieux (porte min_ev = 0.02), val plate à la bande du prédicteur-constant
dès le 25e pas, et ce pour les 10 configs wd x archi. Le couple
architecture (conv 160k params ALÉATOIRES) x données (26k cibles
distinctes, G_t constant par manche) n'extrait RIEN.

Ce que le verdict v2.2 ne règle PAS (donc reste à mesurer, pas à deviner) :

  SONDE 1 — BACKBONE PRÉ-ENTRAÎNÉ. Le goulet est la richesse des
  features, pas la taille de la tête : 48k params entraînables sur des
  features SL (128x6, apprises sur des millions de parties humaines)
  n'ont pas le même problème que 160k params aléatoires. BackboneCritic
  = tronc SL figé (canaux normaux, BN en running-stats) + branche oracle
  entraînable (4 canaux : 3 mains adverses + mur) + tête valeur. Seuls
  la branche et la tête s'entraînent.

  SONDE 2 — CIBLE r_k (delta de manche) au lieu de G_t. G_t contient la
  somme escomptée des manches FUTURES : du point de vue de s, c'est du
  pur bruit (Var(G) ~ 50). Le reward de la manche courante (Var plus
  faible) est la partie que l'état peut expliquer — la mi-manche, les
  mains adverses (oracle) et le mur sont visibles.

Grille par défaut : {std, backbone} x {return, reward} x {0, 3e-4} =
8 fits, quelques minutes chacune sur GPU une fois les données en cache.

CACHE : la collecte (~1 h sur RTX 2060) est la partie chère. Elle est
sauvegardée en npz (--cache, déf. checkpoints/diag_cache.npz) et
rechargée telle quelle aux exécutions suivantes — les sondes suivantes
repartent en quelques minutes, sans re-jeuer un seul env.

Exécution :
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
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong_ai.rl.agent import PPOAgent
from mahjong_ai.rl.rollout import VecRunner
from mahjong_ai.agents.oracle_agent import Critic


# ---------------------------------------------------------------------- #
# Variantes de critic — interface commune forward(obs_n, obs_of) -> (B,)
# en unités NORMALISÉES (cible = G/10), comme le Critic du trainer.
# ---------------------------------------------------------------------- #
class StdCritic(nn.Module):
    """Adaptateur du Critic standard (référence = config v2.2)."""

    def __init__(self, channels=64, n_blocks=4):
        super().__init__()
        self.core = Critic(channels=channels, n_blocks=n_blocks)

    def forward(self, obs_n, obs_of):
        return self.core(torch.cat([obs_n, obs_of], dim=1))


class BackboneCritic(nn.Module):
    """V(s) sur tronc SL pré-entraîné (FIGÉ) + branche oracle entraînable.

    Tronc : celui du modèle discard SL (128x6, canaux normaux 279) —
    features apprises sur des millions de parties humaines, BN en
    running-stats (déterministe). Aucun paramètre du tronc ne bouge.

    Branche oracle : 2 conv (3,1) sur les 4 canaux (3 mains adverses +
    mur) — le privilège asymétrique, la seule partie qui voit la main
    des autres. Tête : mean+max pooling des deux voies -> MLP -> 1.
    Tête finale zero-init : V = 0 exactement au départ (même philosophie
    que le Critic standard).

    Entraînable : ~50k params (branche + tête). Figé : le tronc SL.
    """

    def __init__(self, sl_model, branch_ch=32, hidden=128):
        super().__init__()
        self.trunk = sl_model.trunk
        self.trunk.eval()
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        c = self.trunk.input_conv.out_channels

        self.branch = nn.Sequential(
            nn.Conv2d(4, branch_ch, (3, 1), padding=(1, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch, branch_ch, (3, 1), padding=(1, 0),
                      bias=True),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * c + 2 * branch_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, obs_n, obs_of):
        with torch.no_grad():                       # tronc figé
            f = self.trunk(obs_n.unsqueeze(-1)).squeeze(-1)   # (B,128,34)
        g = self.branch(obs_of.unsqueeze(-1)).squeeze(-1)     # (B,32,34)
        pf = torch.cat([f.mean(-1), f.amax(-1)], dim=1)       # (B,256)
        pg = torch.cat([g.mean(-1), g.amax(-1)], dim=1)       # (B,64)
        return self.head(torch.cat([pf, pg], dim=1)).squeeze(-1)


def build_model_for(variant, agent, device, seed):
    """Construit la variante (partie aléatoire reproductible)."""
    torch.manual_seed(seed)
    if variant == 'std':
        m = StdCritic(channels=64, n_blocks=4)
    elif variant == 'backbone':
        m = BackboneCritic(agent.sl_discard)
    else:
        raise ValueError(variant)
    return m.to(device).eval()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.
                                RawDescriptionHelpFormatter)
    p.add_argument('--sl-discard', required=True)
    p.add_argument('--sl-frozen', required=True)
    p.add_argument('--grp-checkpoint', default=None)
    # collecte (conditions du trainer)
    p.add_argument('--n-envs', type=int, default=16)
    p.add_argument('--batch-games', type=int, default=32)
    p.add_argument('--n-batches', type=int, default=20)
    p.add_argument('--train-batches', type=int, default=12)
    p.add_argument('--val-batches', type=int, default=4)
    p.add_argument('--compute-lookahead', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--cache', default='checkpoints/diag_cache.npz',
                   help='npz de cache des batches collectés. Si le fichier '
                        'existe : chargé tel quel (collection sautée, '
                        '~1 h économisée). Sinon : collecte puis save.')
    # sweep
    p.add_argument('--variants', default='std,backbone')
    p.add_argument('--targets', default='return,reward',
                   help="return = G_t (comme le trainer) ; reward = r_k "
                        "(delta de la manche courante, bruit futur en moins)")
    p.add_argument('--wd-list', default='0,0.0003')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--minibatch', type=int, default=512)
    p.add_argument('--check-every', type=int, default=25)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--critic-lr', type=float, default=1e-3)
    p.add_argument('--reward-scale', type=float, default=10.0)
    return p


# ---------------------------------------------------------------------- #
# Cache des batches
# ---------------------------------------------------------------------- #
def _batch_from_transitions(trs):
    return {
        'obs_n': np.stack([t['obs_normal'] for t in trs]).astype(np.float16),
        'obs_of': np.stack([t['obs_oracle_full'] for t in trs])
            .astype(np.float16),
        'ret': np.array([t['return'] for t in trs], dtype=np.float32),
        'rew': np.array([t['reward'] for t in trs], dtype=np.float32),
    }


def save_cache(path, batches):
    arrays = {}
    for i, b in enumerate(batches):
        arrays[f'b{i}_obs_n'] = b['obs_n']
        arrays[f'b{i}_obs_of'] = b['obs_of']
        arrays[f'b{i}_ret'] = b['ret']
        arrays[f'b{i}_rew'] = b['rew']
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, **arrays)
    mb = os.path.getsize(path) / 1e6
    print(f"[diag] cache sauvé : {path} ({mb:.0f} Mo, "
          f"{len(batches)} batches)", flush=True)


def load_cache(path):
    z = np.load(path)
    n = sum(1 for k in z.files if k.endswith('_ret'))
    batches = []
    for i in range(n):
        batches.append({
            'obs_n': np.asarray(z[f'b{i}_obs_n']),
            'obs_of': np.asarray(z[f'b{i}_obs_of']),
            'ret': np.asarray(z[f'b{i}_ret']),
            'rew': np.asarray(z[f'b{i}_rew']),
        })
    print(f"[diag] cache chargé : {path} ({n} batches, collection sautée)",
          flush=True)
    return batches


def _distinct(vals):
    return len(np.unique(np.round(np.asarray(vals, dtype=np.float64), 3)))


# ---------------------------------------------------------------------- #
# Fits
# ---------------------------------------------------------------------- #
def _values(model, obs_n, obs_of, device, rs, bs=2048):
    """V(s) en unités reward, par tranches (aucun gradient)."""
    out = []
    with torch.no_grad():
        for s in range(0, len(obs_n), bs):
            on = torch.as_tensor(np.asarray(obs_n[s:s + bs]),
                                 dtype=torch.float32, device=device)
            oof = torch.as_tensor(np.asarray(obs_of[s:s + bs]),
                                  dtype=torch.float32, device=device)
            out.append(model(on, oof).cpu().numpy())
    return np.concatenate(out) * rs


def _eval_loss(model, obs_n, obs_of, target, device, rs):
    """0.5*MSE normalisée (comparable à vl / cfit / cval)."""
    v = _values(model, obs_n, obs_of, device, rs) / rs
    return float(0.5 * ((v - target / rs) ** 2).mean())


def fit_one(model, tr, va, test_batches, key, wd, args, device):
    """Un fit complet ; époque sélectionnée sur val (early-stop)."""
    rng = np.random.default_rng(args.seed)
    tn = np.concatenate([b['obs_n'] for b in tr])
    to = np.concatenate([b['obs_of'] for b in tr])
    tt = np.concatenate([b[key] for b in tr])
    vn = np.concatenate([b['obs_n'] for b in va])
    vo = np.concatenate([b['obs_of'] for b in va])
    vt = np.concatenate([b[key] for b in va])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.critic_lr)
    n_train = len(tt)

    best_val, best_state, since_best, best_step, stopped_at = (
        None, None, 0, 0, None)
    rs = args.reward_scale
    it = 0
    for it in range(1, args.steps + 1):
        idx = rng.integers(0, n_train, size=args.minibatch)
        on = torch.as_tensor(tn[idx], dtype=torch.float32, device=device)
        oof = torch.as_tensor(to[idx], dtype=torch.float32, device=device)
        ret = torch.as_tensor(tt[idx], dtype=torch.float32,
                              device=device) / rs
        v = model(on, oof)
        loss = 0.5 * ((v - ret) ** 2).mean()
        if wd > 0:
            loss = loss + wd * sum(torch.sum(p * p) for p in params)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % args.check_every == 0:
            vloss = _eval_loss(model, vn, vo, vt, device, rs)
            if best_val is None or vloss < best_val - 1e-6:
                best_val, since_best, best_step = vloss, 0, it
                best_state = {k: w.detach().cpu().clone()
                              for k, w in model.state_dict().items()}
            else:
                since_best += 1
                if since_best >= args.patience:
                    stopped_at = it
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    evs, losses = [], []
    for tb in test_batches:
        G = tb[key]
        v = _values(model, tb['obs_n'], tb['obs_of'], device, rs)
        evs.append(1.0 - float((G - v).var()) / max(float(G.var()), 1e-8))
        losses.append(float(0.5 * ((v / rs - G / rs) ** 2).mean()))
    return {'best_val': best_val, 'best_step': best_step,
            'stopped_at': stopped_at, 'n_steps': it,
            'evs': evs, 'ev_mean': float(np.mean(evs)),
            'ev_min': float(np.min(evs)), 'losses': losses}


# ---------------------------------------------------------------------- #
def main():
    args = build_parser().parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"[diag] GPU : {torch.cuda.get_device_name(0)}", flush=True)
    torch.set_num_threads(args.torch_threads)

    agent = PPOAgent(args.sl_discard, args.sl_frozen, device=device,
                     critic_lr=args.critic_lr)

    # ---------------- données : cache ou collecte ---------------- #
    if os.path.isfile(args.cache):
        batches = load_cache(args.cache)
    else:
        print(f"[diag] protocole : {args.n_batches} batches "
              f"({args.train_batches} train / {args.val_batches} val / "
              f"{args.n_batches - args.train_batches - args.val_batches} "
              f"test), policy = SL figée, gamma_t=0, mode=opponent",
              flush=True)
        runner = VecRunner(
            agent, n_envs=args.n_envs, seed=args.seed,
            grp_path=args.grp_checkpoint,
            compute_lookahead=args.compute_lookahead, danger_labels=True,
            mode='opponent', device=device)
        batches = []
        t0 = time.time()
        for i in range(args.n_batches):
            eps, _ = runner.collect_batch(args.batch_games, gamma_t=0.0)
            b = _batch_from_transitions(
                [t for ep in eps for t in ep])
            batches.append(b)
            print(f"  [{i+1:2d}/{args.n_batches}] {len(b['ret'])} tr, "
                  f"G distincts={_distinct(b['ret'])} "
                  f"r_k distincts={_distinct(b['rew'])} "
                  f"(G var={b['ret'].var():.1f} r_k var={b['rew'].var():.1f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
        save_cache(args.cache, batches)

    n_tr, n_va = args.train_batches, args.val_batches
    need = n_tr + n_va + 2
    assert len(batches) >= need, \
        f"cache/collecte insuffisant : {len(batches)} < {need} " \
        f"(relancer avec --n-batches {need} et supprimer {args.cache})"
    tr, va = batches[:n_tr], batches[n_tr:n_tr + n_va]
    test_batches = batches[n_tr + n_va:]
    print(f"[diag] splits : train={sum(len(b['ret']) for b in tr)} "
          f"val={sum(len(b['ret']) for b in va)} "
          f"test={sum(len(b['ret']) for b in test_batches)} tr", flush=True)

    # ---------------- références ---------------- #
    print("\n[diag] références (test) :")
    train_means = {}
    for key, label in (('ret', 'G_t'), ('rew', 'r_k')):
        tm = float(np.mean([x for b in tr for x in b[key]]))
        train_means[key] = tm
        for k, tb in enumerate(test_batches):
            G = tb[key]
            print(f"  test[{k}] {label:3s}: n={len(G)}, "
                  f"Var={G.var():.1f}, EV(const)=+0.000, "
                  f"EV(moy-train)={1.0 - float((G - tm).var()) / max(G.var(), 1e-8):+.3f}, "
                  f"loss(const)={float(0.5 * G.var()) / args.reward_scale ** 2:.3f}",
                  flush=True)

    # ---------------- sweep ---------------- #
    variants = [v.strip() for v in args.variants.split(',')]
    targets = [('ret', 'return') if t.strip() == 'return'
               else ('rew', 'reward') for t in args.targets.split(',')]
    wds = [float(x) for x in args.wd_list.split(',')]
    print(f"\n[diag] sweep : variant x {variants} | target x "
          f"{[lbl for _, lbl in targets]} | wd x {wds} | "
          f"budget {args.steps} steps, early-stop (val, patience "
          f"{args.patience})\n", flush=True)

    rows = []
    for variant in variants:
        for key, tname in targets:
            for wd in wds:
                model = build_model_for(variant, agent, device, args.seed)
                r = fit_one(model, tr, va, test_batches, key, wd,
                            args, device)
                rows.append((variant, tname, wd, r))
                print(f"  {variant:9s} target={tname:6s} wd={wd:<7g} : "
                      f"val={r['best_val']:.3f}@{r['best_step']} | "
                      f"EV test min/mean = {r['ev_min']:+.3f}/{r['ev_mean']:+.3f}"
                      f" | loss test {np.mean(r['losses']):.3f}", flush=True)

    # ---------------- verdict ---------------- #
    best = max(rows, key=lambda x: x[3]['ev_mean'])
    v_b, t_b, wd_b, r_b = best
    print("\n" + "=" * 64)
    print(f"[VERDICT] meilleure config : {v_b} + target {t_b} + wd={wd_b:g}"
          f" -> EV test moyen {r_b['ev_mean']:+.3f} (min {r_b['ev_min']:+.3f})")
    if r_b['ev_mean'] >= 0.10:
        print("[VERDICT] GO : le signal existe, la sonde le mesure.")
        print(f"          Rapporter ce tableau — le patch v2.4 câblera "
              f"«{v_b}» et la cible «{t_b}»")
        print("          dans le trainer (compute_advantages / critic).")
        print("          NE PAS relancer le trainer avant ce câblage.")
    elif r_b['ev_mean'] >= 0.02:
        print("[VERDICT] MARGINAL (la porte min_ev=0.02 passera rarement).")
        print("          Rapporter le tableau avant toute décision.")
    else:
        print("[VERDICT] NON-PRÉDICTIBLE même avec backbone et cible r_k.")
        print("          À cette échelle, la réduction de variance par V(s)")
        print("          n'est PAS disponible en régime no-oracle — comme")
        print("          l'ablation du papier Suphx le laissait craindre.")
        print("          Options avant de relancer quoi que ce soit :")
        print("            (a) curriculum oracle (voie du papier) — corriger")
        print("                d'abord les pièges du pilot (PHASE_POLICY_LR")
        print("                phase 3 = 0.1 encore en dur) ;")
        print("            (b) accepter batch_mean + ancre KL 0.2, run long")
        print("                sans attente de gain mesurable ;")
        print("            (c) améliorer le SL d'abord (val_acc 0.667 = le")
        print("                vrai plafond actuel).")
        print("          Discuter AVANT de relancer.")
    print("=" * 64)


if __name__ == '__main__':
    main()
