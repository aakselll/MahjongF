"""[V2.4] Le critic peut-il prédire la valeur ? Troisième salve de sondes.

Verdict v2.3 (policy SL figée, 20 batches, champ production rl_defense2) :
std + target r_k + wd=3e-4 -> EV test +0.033 (min -0.013) : MARGINAL.
Le run rl_defense2 (return-rho=0, G_t = r_k, adv=batch_mean) A ÉCHOUÉ
sur ce signal : mr échantillonné 2.35 -> 2.67, argmax 2.53 (figé), proxy
myopique ret_mean -0.14 -> -0.87. Conclusion de session : le goulot
n'est pas la variance du gradient, c'est l'INFORMATION PAR TRANSITION.

NOUVEAUTÉ v2.4 — cibles par COMPOSANTE (jamais testées) :
  ghand     to-go dans la manche courante, coupé à la fin de CETTE
            manche, attribution causale par événement (cf.
            diag_reward_structure pour la définition exacte).
            NB : sous l'attribution plateau-par-main, ghand ≡ r_k à
            ~1e-2 près — le rapport de structure le chiffre ; le fit
            n'est là que pour l'inscrire sur table.
  ghand_nc  ghand sans la correction terminale de rang.
  rnd       le delta GRP de la manche, SEUL — la composante qui domine
            r_k. La vraie question nouvelle : un critic peut-il prédire
            le delta de main DEPUIS l'état ? Si oui (EV >= 0.10), une
            baseline ÉTAT-DÉPENDANTE existe pour la partie plateau de
            l'avantage ; sinon le critic est mort en régime no-oracle,
            et le plan D bascule sur « événements seuls » (masque du
            policy loss sur les transitions non informatives).

PROTOCOLE : inchangé v2.3 (même grille std/backbone, mêmes pas, même
early-stop) pour que les EV restent comparables entre v2.3 et v2.4.

DONNÉES : le cache v2.3 ne contient que ret/rew. Les nouvelles cibles
exigent le cache INSTRUMENTÉ de diag_reward_structure (clés crr/pen/
rnd/fls/cor/ghd/ghn/round). Lancer d'abord (collecte ~1 h) :
    py -3.14 -m mahjong_ai.diag_reward_structure ^
        --sl-discard ... --sl-frozen ... ^
        --grp-checkpoint checkpoints/grp_tonpusen.pt ^
        --compute-lookahead --n-batches 20 ^
        --cache checkpoints/diag_struct_cache.npz
puis le présent diag avec le MÊME --cache. Un cache v2.3 (ret/rew
seuls) continue de fonctionner avec --targets return,reward.

Exécution (fits seuls, ~3 min par config sur GPU, ~30 min pour 12) :
    py -3.14 -m mahjong_ai.diag_critic_learnability ^
        --sl-discard checkpoints/sl_tonpusen_la/discard.pt ^
        --sl-frozen checkpoints/sl_tonpusen_la ^
        --cache checkpoints/diag_struct_cache.npz ^
        --targets reward,rnd,ghand_nc --variants std,backbone
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong_ai.rl.agent import PPOAgent
from mahjong_ai.agents.oracle_agent import Critic
from mahjong_ai.diag_reward_structure import load_cache


# ---------------------------------------------------------------------- #
# Variantes de critic — interface commune forward(obs_n, obs_of) -> (B,)
# en unités NORMALISÉES (cible = G/10), comme le Critic du trainer.
# ---------------------------------------------------------------------- #
class StdCritic(nn.Module):
    """Adaptateur du Critic standard (référence = config v2.2/v2.3)."""

    def __init__(self, channels=64, n_blocks=4):
        super().__init__()
        self.core = Critic(channels=channels, n_blocks=n_blocks)

    def forward(self, obs_n, obs_of):
        return self.core(torch.cat([obs_n, obs_of], dim=1))


class BackboneCritic(nn.Module):
    """V(s) sur tronc SL pré-entraîné (FIGÉ) + branche oracle entraînable.

    Tronc : celui du modèle discard SL (canaux normaux 279) — features
    apprises sur des millions de parties humaines, BN en running-stats
    (déterministe). Aucun paramètre du tronc ne bouge.

    Branche oracle : 2 conv (3,1) sur les 4 canaux (3 mains adverses +
    mur). Tête : mean+max pooling des deux voies -> MLP -> 1.
    Tête finale zero-init : V = 0 exactement au départ.
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


# ---------------------------------------------------------------------- #
# Cibles : nom CLI -> (clé du batch, label)
# ---------------------------------------------------------------------- #
TARGETS = {
    'return':   ('ret', 'G_t(rho)'),
    'reward':   ('rew', 'r_k'),
    'ghand':    ('ghd', 'ghand'),
    'ghand_nc': ('ghn', 'ghand_nc'),
    'rnd':      ('rnd', 'rnd(delta main)'),
}
# ordre canonique d'affichage des références (présentes uniquement)
CANON = ('ret', 'rew', 'ghd', 'ghn', 'rnd')
CANON_LABEL = {'ret': 'G_t(rho)', 'rew': 'r_k', 'ghd': 'ghand',
               'ghn': 'ghand_nc', 'rnd': 'rnd(delta main)'}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.
                                RawDescriptionHelpFormatter)
    p.add_argument('--sl-discard', required=True)
    p.add_argument('--sl-frozen', required=True)
    p.add_argument('--cache', required=True,
                   help='npz de diag_reward_structure (étendu) ou du '
                        'v2.3 (ret/rew seuls)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--torch-threads', type=int, default=2)
    # sweep
    p.add_argument('--variants', default='std,backbone')
    p.add_argument('--targets', default='return,reward',
                   help='return = G_t(rho) ; reward = r_k ; ghand = to-go '
                        'dans la manche courante ; ghand_nc = sans '
                        'correction ; rnd = delta de manche seul. '
                        'Les 3 dernières exigent le cache instrumenté.')
    p.add_argument('--wd-list', default='0,0.0003')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--minibatch', type=int, default=512)
    p.add_argument('--check-every', type=int, default=25)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--critic-lr', type=float, default=1e-3)
    p.add_argument('--reward-scale', type=float, default=10.0)
    return p


# ---------------------------------------------------------------------- #
# Fits (identiques v2.3 : comparabilité)
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

    # ---------------- données : cache instrumenté requis ---------------- #
    if not os.path.isfile(args.cache):
        raise SystemExit(
            f"[diag] cache introuvable : {args.cache}\n"
            f"        v2.4 ne collecte plus : lancer d'abord "
            f"diag_reward_structure\n"
            f"        (il construit ce cache en ~1 h).")
    batches = load_cache(args.cache)

    # splits : 12/4/4 sur 20 batches (v2.3) ; sinon repli proportionnel
    n_batches = len(batches)
    if n_batches >= 20:
        n_tr, n_va = 12, 4
    else:
        n_va = max(1, min(int(0.2 * n_batches), n_batches - 3))
        n_tr = max(1, min(int(0.6 * n_batches), n_batches - n_va - 2))
    need = n_tr + n_va + 2
    assert n_batches >= need, \
        f"cache insuffisant : {n_batches} < {need} (relancer la collecte " \
        f"avec --n-batches 20)"
    tr, va = batches[:n_tr], batches[n_tr:n_tr + n_va]
    test_batches = batches[n_tr + n_va:]
    print(f"[diag] splits : train={sum(len(b['rew']) for b in tr)} "
          f"val={sum(len(b['rew']) for b in va)} "
          f"test={sum(len(b['rew']) for b in test_batches)} tr", flush=True)

    # ---------------- cibles demandées : disponibles ? ---------------- #
    wanted = []
    for t in args.targets.split(','):
        t = t.strip()
        if t not in TARGETS:
            raise SystemExit(f"[diag] cible inconnue : {t!r} "
                              f"(choix : {', '.join(TARGETS)})")
        wanted.append((t, *TARGETS[t]))
    present = set(k for k in CANON if all(k in b for b in batches))
    missing = [k for _, k, _ in wanted if k not in present]
    if missing:
        raise SystemExit(
            f"[diag] cible(s) absente(s) du cache : {missing}\n"
            f"        ce cache ne contient que {sorted(present)}. Lancer "
            f"diag_reward_structure (cache étendu) ou restreindre "
            f"--targets à : {', '.join(n for n, (k, _) in
                                        TARGETS.items() if k in present)}")

    # ---------------- références ---------------- #
    print("\n[diag] références (test) :")
    train_means = {}
    for key in CANON:
        if key not in present:
            continue
        tm = float(np.mean([x for b in tr for x in b[key]]))
        train_means[key] = tm
        for k, tb in enumerate(test_batches):
            G = tb[key]
            print(f"  test[{k}] {CANON_LABEL[key]:12s}: n={len(G)}, "
                  f"Var={G.var():.1f}, EV(const)=+0.000, "
                  f"EV(moy-train)="
                  f"{1.0 - float((G - tm).var()) / max(G.var(), 1e-8):+.3f},"
                  f" loss(const)="
                  f"{float(0.5 * G.var()) / args.reward_scale ** 2:.3f}",
                  flush=True)

    # ---------------- sweep ---------------- #
    variants = [v.strip() for v in args.variants.split(',')]
    wds = [float(x) for x in args.wd_list.split(',')]
    print(f"\n[diag] sweep : variant x {variants} | target x "
          f"{[t for t, _, _ in wanted]} | wd x {wds} | "
          f"budget {args.steps} steps, early-stop (val, patience "
          f"{args.patience})\n", flush=True)

    rows = []
    for variant in variants:
        for tname, key, _lbl in wanted:
            for wd in wds:
                model = build_model_for(variant, agent, device, args.seed)
                r = fit_one(model, tr, va, test_batches, key, wd,
                            args, device)
                rows.append((variant, tname, wd, r))
                print(f"  {variant:9s} target={tname:9s} wd={wd:<7g} : "
                      f"val={r['best_val']:.3f}@{r['best_step']} | "
                      f"EV test min/mean = {r['ev_min']:+.3f}/"
                      f"{r['ev_mean']:+.3f}"
                      f" | loss test {np.mean(r['losses']):.3f}", flush=True)

    # ---------------- verdict ---------------- #
    best = max(rows, key=lambda x: x[3]['ev_mean'])
    v_b, t_b, wd_b, r_b = best
    print("\n" + "=" * 64)
    print(f"[VERDICT] meilleure config : {v_b} + target {t_b} + "
          f"wd={wd_b:g} -> EV test moyen {r_b['ev_mean']:+.3f} "
          f"(min {r_b['ev_min']:+.3f})")
    if r_b['ev_mean'] >= 0.10:
        print("[VERDICT] GO : le signal existe, la sonde le mesure.")
        print(f"          Rapporter ce tableau + le VERDICT STRUCTURE de "
              f"diag_reward_structure : le patch v2.4 câblera «{v_b}» et "
              f"la cible «{t_b}» comme baseline état-dépendante dans "
              f"compute_advantages/critic (plan D, branche critic).")
        print("          NE PAS relancer le trainer avant ce câblage.")
    elif r_b['ev_mean'] >= 0.02:
        print("[VERDICT] MARGINAL (la porte min_ev=0.02 passera rarement).")
        print("          Rapporter le tableau avant toute décision :")
        print("          en l'état, le plan D bascule sur « événements "
              "seuls » (masque du policy loss sur les transitions non "
              "informatives).")
    else:
        print("[VERDICT] NON-PRÉDICTIBLE, toutes cibles confondues.")
        print("          Le critic est mort en régime no-oracle : la "
              "réduction de variance par V(s) n'est PAS "
              "disponible à cette échelle.")
        print("          Plan D, branche « événements seuls » : masquer "
              "le policy loss sur les transitions non informatives, "
              "n'apprendre que sur les événements (pénalités + correction "
              "terminale).")
        print("          Rapporter ce tableau AVANT de relancer quoi que "
              "ce soit.")
    print("=" * 64)


if __name__ == '__main__':
    main()
