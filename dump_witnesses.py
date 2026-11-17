# -*- coding: utf-8 -*-
"""dump_witnesses.py -- dump BRUT des enregistrements sweep / npz / summary
pour des donnees temoins, afin d'identifier ce que la colonne 'pts' du sweep
mesure vraiment (cf. reconciliation v2d vs diag_v3).

Usage (chemins par defaut = ceux de reconcile_paid.py) :
    py -3.14 dump_witnesses.py
    py -3.14 dump_witnesses.py --games 1:21,1:14
"""
import argparse
import json
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

AP = argparse.ArgumentParser()
AP.add_argument("--npz", default=r"checkpoints\danger_diag_v3\diag_dealins.npz")
AP.add_argument("--results", default=r"checkpoints\danger_blend_sweep_v2d\results.json")
AP.add_argument("--summary", default=r"checkpoints\danger_diag_v3\diag_summary.json")
AP.add_argument("--arm", default="baseline")
AP.add_argument("--games", default="1:21,1:14,1:2,2:2,1:12,2:5,2:15,2:18,2:28,2:30",
                help="liste chunk:env separee par des virgules")
A = AP.parse_args()

WANT = []
for g in A.games.split(","):
    c, e = g.strip().split(":")
    WANT.append((int(c), int(e)))

MAXCHAR = 6000


def hr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def show(obj):
    s = json.dumps(obj, ensure_ascii=False, indent=1, default=str)
    print(s[:MAXCHAR] + ("\n...[tronque]" if len(s) > MAXCHAR else ""))


def scalar(v):
    try:
        return v.item()
    except Exception:
        try:
            return v.tolist()
        except Exception:
            return repr(v)


# --------------------------------------------------------------------------
hr("[0] results.json -- structure globale (cote sweep)")
res = None
try:
    with open(A.results, encoding="utf-8") as f:
        res = json.load(f)
    if isinstance(res, dict):
        print("cles racine :", list(res.keys()))
    else:
        print("type racine :", type(res).__name__, "| longueur :", len(res))
except Exception as ex:
    print("ERREUR lecture results.json :", repr(ex))

# --------------------------------------------------------------------------
hr("[1] results.json -- enregistrements BRUTS du bras '%s' pour les temoins" % A.arm)


def find_games(obj, path="root"):
    """Cherche defensivelement la liste de donnees d'un bras."""
    if isinstance(obj, dict):
        for key in ("arms", A.arm):
            if key in obj:
                r = find_games(obj[key], path + "." + key)
                if r:
                    return r
        for key in ("games", "runs", "results", "records", "episodes"):
            if key in obj:
                r = find_games(obj[key], path + "." + key)
                if r:
                    return r
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        if "arm" in obj[0]:
            sel = [r for r in obj if r.get("arm") == A.arm]
            if sel:
                return path + "[arm=%s]" % A.arm, sel
        return path, obj
    return None


if res is not None:
    found = find_games(res)
    if not found:
        print("!! structure inconnue -> colle-moi la sortie de [0] entierement")
    else:
        path, games = found
        print("liste trouvee a : %s | %d enregistrements" % (path, len(games)))
        print("champs d'un enregistrement :", sorted(games[0].keys()))
        kc = next((k for k in ("chunk",) if k in games[0]), None)
        ke = next((k for k in ("env_idx", "env", "envi") if k in games[0]), None)
        hits = 0
        for rec in games:
            if kc and ke and (rec[kc], rec[ke]) in WANT:
                hits += 1
                print("\n--- sweep BRUT chunk %s env %s ---" % (rec[kc], rec[ke]))
                show(rec)
        if hits == 0:
            print("!! temoins non trouves avec chunk=%s env=%s ; 2 premiers enregistrements :"
                  % (kc, ke))
            for rec in games[:2]:
                show(rec)

# --------------------------------------------------------------------------
hr("[2] npz -- champs disponibles + lignes marquees des donnees temoins")
try:
    z = np.load(A.npz, allow_pickle=True)
    names = list(z.files)
    print("arrays :", names)
    ccol = next((n for n in ("chunk",) if n in names), None)
    ecol = next((n for n in ("env_idx", "env", "envi") if n in names), None)
    rcol = next((n for n in ("round", "rnd", "ronde", "kyoku") if n in names), None)
    flag_cols = [n for n in names if any(t in n.lower() for t in
                 ("lbl", "label", "paid", "paye", "stick", "chankan", "riichi",
                  "win", "tile", "seat", "pay", "pts"))]
    print("colonnes id : chunk=%s env=%s round=%s" % (ccol, ecol, rcol))
    print("colonnes potentiellement utiles :", flag_cols)
    if ccol and ecol:
        ch, ev = z[ccol], z[ecol]
        for (c, e) in WANT:
            idx = np.nonzero((ch == c) & (ev == e))[0]
            print("\n--- npz chunk %d env %d : %d lignes ---" % (c, e, len(idx)))
            keep = set()
            if rcol is not None:
                last = {}
                for i in idx:
                    last[int(scalar(z[rcol][i]))] = int(i)
                keep.update(last.values())  # derniere deffaude de chaque round
            for n in flag_cols:
                try:
                    arr = z[n]
                    if arr.ndim == 1 and arr.dtype.kind in "biuf":
                        for i in idx:
                            if float(arr[i]) != 0:
                                keep.add(int(i))
                except Exception:
                    pass
            for i in sorted(keep):
                d = {n: scalar(z[n][i]) for n in names}
                print(d)
except Exception as ex:
    print("ERREUR npz :", repr(ex))

# --------------------------------------------------------------------------
hr("[3] diag_summary.json -- rounds BRUTS (tous les champs) des temoins")
try:
    with open(A.summary, encoding="utf-8") as f:
        summ = json.load(f)
    recs = []

    def walk(o):
        if isinstance(o, list):
            for x in o:
                if isinstance(x, dict) and "round" in x and "chunk" in x:
                    recs.append(x)
                else:
                    walk(x)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)

    walk(summ)
    print("%d rounds trouves au total" % len(recs))
    if recs:
        print("champs d'un round :", sorted(recs[0].keys()))
    for (c, e) in WANT:
        sel = [r for r in recs if r.get("chunk") == c
               and r.get("env_idx", r.get("env")) == e]
        if sel:
            print("\n--- summary chunk %d env %d ---" % (c, e))
            for r in sorted(sel, key=lambda r: r.get("round", 0)):
                print(json.dumps(r, ensure_ascii=False, default=str))
except Exception as ex:
    print("ERREUR summary :", repr(ex))

# --------------------------------------------------------------------------
hr("CE QU'ON CHERCHE EN LISANT CE DUMP")
print("""1) Dans [1] : l'enregistrement sweep contient-il une liste d'evenements
   (paiements par deal-in) ou seulement des agregats ? Si des montants
   apparaissent, les comparer aux rounds de [3] : s'ils n'existent dans AUCUN
   round de la meme donnee, le canal 'pts' du sweep n'est pas celui qu'on croit
   (mauvais index de step, mauvais siege, somme multi-bras, multi-ron double...).
2) Dans [2], pour les donnees 1:12, 2:5, 2:15, 2:18, 2:28, 2:30 (payes npz sans
   equivalent sweep) : chercher un champ riichi / tuile gagnante / payeur.
   - agent riichi au round du PAYE, ou tuile gagnante != tuile de la ligne
     -> le npz attribue a tort (perte de stick / pao) ;
   - tuile gagnante == tuile de la ligne -> vrai deal-in sur deffausse forcee
     que le sweep n'exclut que par convention (a trancher pour la porte v2e).
3) Dans [3] : des champs absents du print de reconcile_paid.py (riichi,
   tuile gagnante, paiements par siege) qui permettraient de trancher.""")
